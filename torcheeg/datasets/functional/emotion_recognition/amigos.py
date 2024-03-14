import os
import re
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, List, Union

import scipy.io as scio
from torcheeg.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024


def transform_producer(file_name: str, root_path: str, chunk_size: int,
                       overlap: int, num_channel: int, num_trial: int,
                       skipped_subjects: List, num_baseline: int,
                       baseline_chunk_size: int, before_trial: Union[Callable,
                                                                     None],
                       transform: Union[Callable, None],
                       after_trial: Union[Callable, None], queue: Queue):
    subject = int(re.findall(r'Data_Preprocessed_P(\d*).mat',
                             file_name)[0])  # subject (40)

    if subject in skipped_subjects:
        return

    data = scio.loadmat(os.path.join(root_path, file_name),
                        verify_compressed_data_integrity=False)
    samples = data['joined_data'][
        0]  # trial (20), timestep(n*128), channel(17) (14 channels are EEGs)
    # label file
    labels = data['labels_selfassessment'][
        0]  # trial (20), label of different dimensions ((1, 12))

    write_pointer = 0

    max_len = len(samples)
    if not (num_trial <= 0):
        max_len = min(len(samples), num_trial)

    # loop for each trial
    for trial_id in range(max_len):
        # extract baseline signals
        trial_samples = samples[trial_id]

        # record the common meta info
        trial_meta_info = {'subject_id': subject, 'trial_id': trial_id}
        trial_rating = labels[trial_id][0]  # label of different dimensions (12)

        # missing values
        if (not sum(trial_samples.shape)) or (not sum(trial_rating.shape)):
            # 3 of the participants (08,24,28<->32) of the previous experiment did not watch a set of 4 long affective
            if sum(trial_samples.shape) != sum(trial_rating.shape):
                print(
                    f'[WARNING] Find EEG signals without labels, or labels without EEG signals. Please check the {trial_id + 1}-th experiment of the {subject}-th subject in the file {file_name}. TorchEEG currently skipped the mismatched data.'
                )
            continue

        trial_samples = trial_samples.swapaxes(
            1, 0)  # channel(17), timestep(n*128)
        if before_trial:
            trial_samples = before_trial(trial_samples)

        for label_index, label_name in enumerate([
                'arousal', 'valence', 'dominance', 'liking', 'familiarity',
                'neutral', 'disgust', 'happiness', 'surprise', 'anger', 'fear',
                'sadness'
        ]):
            trial_meta_info[label_name] = trial_rating[label_index]

        # extract baseline signals
        trial_baseline_sample = trial_samples[:num_channel, :
                                              baseline_chunk_size *
                                              num_baseline]  # channel(14), timestep(5*128)
        trial_baseline_sample = trial_baseline_sample.reshape(
            num_channel, num_baseline,
            baseline_chunk_size).mean(axis=1)  # channel(14), timestep(128)

        trial_queue = []

        start_at = baseline_chunk_size * num_baseline
        if chunk_size <= 0:
            chunk_size = trial_samples.shape[1] - start_at

        # chunk with chunk size
        end_at = start_at + chunk_size
        # calculate moving step
        step = chunk_size - overlap

        while end_at <= trial_samples.shape[1]:
            clip_sample = trial_samples[:num_channel, start_at:end_at]

            t_eeg = clip_sample
            t_baseline = trial_baseline_sample
            if not transform is None:
                t = transform(eeg=clip_sample, baseline=trial_baseline_sample)
                t_eeg = t['eeg']
                t_baseline = t['baseline']

            # put baseline signal into IO
            if not 'baseline_id' in trial_meta_info:
                trial_base_id = f'{file_name}_{write_pointer}'
                queue.put({'eeg': t_baseline, 'key': trial_base_id})
                write_pointer += 1
                trial_meta_info['baseline_id'] = trial_base_id

            clip_id = f'{file_name}_{write_pointer}'
            write_pointer += 1

            # record meta info for each signal
            record_info = {
                'start_at': start_at,
                'end_at': end_at,
                'clip_id': clip_id
            }
            record_info.update(trial_meta_info)
            if after_trial:
                trial_queue.append({
                    'eeg': t_eeg,
                    'key': clip_id,
                    'info': record_info
                })
            else:
                queue.put({'eeg': t_eeg, 'key': clip_id, 'info': record_info})

            start_at = start_at + step
            end_at = start_at + chunk_size

        if len(trial_queue) and after_trial:
            trial_queue = after_trial(trial_queue)
            for obj in trial_queue:
                assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg`, `key` and `info` as keys.'
                queue.put(obj)


def io_consumer(write_eeg_fn: Callable, write_info_fn: Callable, queue: Queue):
    while True:
        item = queue.get()
        if not item is None:
            eeg = item['eeg']
            key = item['key']
            write_eeg_fn(eeg, key)
            if 'info' in item:
                info = item['info']
                write_info_fn(info)
        else:
            break


class SingleProcessingQueue:
    def __init__(self, write_eeg_fn: Callable, write_info_fn: Callable):
        self.write_eeg_fn = write_eeg_fn
        self.write_info_fn = write_info_fn

    def put(self, item):
        eeg = item['eeg']
        key = item['key']
        self.write_eeg_fn(eeg, key)
        if 'info' in item:
            info = item['info']
            self.write_info_fn(info)


def amigos_constructor(root_path: str = './data_preprocessed',
                       chunk_size: int = 128,
                       overlap: int = 0,
                       num_channel: int = 14,
                       num_trial: int = 16,
                       skipped_subjects: List[int] = [
                           9, 12, 21, 22, 23, 24, 33
                       ],
                       num_baseline: int = 5,
                       baseline_chunk_size: int = 128,
                       before_trial: Union[None, Callable] = None,
                       transform: Union[None, Callable] = None,
                       after_trial: Union[Callable, None] = None,
                       io_path: str = './io/amigos',
                       io_size: int = 10485760,
                       io_mode: str = 'lmdb',
                       num_worker: int = 0,
                       verbose: bool = True) -> None:
    # init IO
    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    if os.path.exists(io_path) and not os.path.getsize(meta_info_io_path) == 0:
        print(
            f'The target folder already exists, if you need to regenerate the database IO, please delete the path {io_path}.'
        )
        return

    os.makedirs(io_path, exist_ok=True)

    info_io = MetaInfoIO(meta_info_io_path)
    eeg_io = EEGSignalIO(eeg_signal_io_path, io_size=io_size, io_mode=io_mode)

    # loop to access the dataset files
    file_list = os.listdir(root_path)

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[AMIGOS]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)

        partial_mp_fn = partial(transform_producer,
                                root_path=root_path,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                num_channel=num_channel,
                                num_trial=num_trial,
                                skipped_subjects=skipped_subjects,
                                num_baseline=num_baseline,
                                baseline_chunk_size=baseline_chunk_size,
                                before_trial=before_trial,
                                transform=transform,
                                after_trial=after_trial,
                                queue=queue)

        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg,
                                            info_io.write_info, queue),
                                      daemon=True)
        io_consumer_process.start()

        for _ in Pool(num_worker).imap(partial_mp_fn, file_list):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for file_name in file_list:
            transform_producer(file_name=file_name,
                               root_path=root_path,
                               chunk_size=chunk_size,
                               overlap=overlap,
                               num_channel=num_channel,
                               num_trial=num_trial,
                               skipped_subjects=skipped_subjects,
                               num_baseline=num_baseline,
                               baseline_chunk_size=baseline_chunk_size,
                               before_trial=before_trial,
                               transform=transform,
                               after_trial=after_trial,
                               queue=SingleProcessingQueue(
                                   eeg_io.write_eeg, info_io.write_info))
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')
