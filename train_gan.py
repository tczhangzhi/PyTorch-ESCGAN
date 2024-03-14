import logging
import math
import os
import random
import time

import nni
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from models import ESCDiscriminator, ESCGenerator, Extractor, Predictor
from torcheeg import transforms
from torcheeg.datasets import (AMIGOSDataset, DEAPDataset,
                               SEEDFeatureDataset)
from torcheeg.datasets.constants.emotion_recognition.amigos import (
    AMIGOS_CHANNEL_LIST, AMIGOS_CHANNEL_LOCATION_DICT)
from torcheeg.datasets.constants.emotion_recognition.deap import (
    DEAP_CHANNEL_LIST, DEAP_CHANNEL_LOCATION_DICT)
from torcheeg.datasets.constants.emotion_recognition.seed import (
    SEED_CHANNEL_LIST, SEED_CHANNEL_LOCATION_DICT)
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.trainers import CGANTrainer
from torcheeg.utils import plot_feature_topomap

TRAIL_ID = nni.get_trial_id()

RECEIVED_PARAMS = {
    'hid_channels': 128,
    'num_classes': 2,
    'lr': 0.0001,
    'generator_lr': 0.0001,
    'discriminator_lr': 0.0001,
    'weight_decay': 0.0001,
    'batch_size': 256,
    'n_splits': 5,
    'num_bands': 5,
    'num_epochs': 100,
    'w_boundary': 5.0,
    'w_diverse': 1.0,
    'w_smooth': 0.9,
    'target': 'emotion',
    'dataset': 'seed'
}

if not TRAIL_ID == 'STANDALONE':
    RECEIVED_PARAMS.update(nni.get_next_parameter())

if RECEIVED_PARAMS['dataset'] == 'deap' and RECEIVED_PARAMS[
        'target'] == 'valence':
    RECEIVED_PARAMS.update({
        "generator_lr": 0.0001,
        "discriminator_lr": 0.0001,
        "weight_decay": 0,
        "w_boundary": 5,
        "w_smooth": 0.85,
        "num_epochs": 100,
        "dataset": "deap",
        "target": "valence"
    })
elif RECEIVED_PARAMS['dataset'] == 'deap' and RECEIVED_PARAMS[
        'target'] == 'arousal':
    RECEIVED_PARAMS.update({
        "generator_lr": 0.0001,
        "discriminator_lr": 0.0001,
        "weight_decay": 0,
        "w_boundary": 5,
        "w_smooth": 0.85,
        "num_epochs": 100,
        "dataset": "deap",
        "target": "arousal"
    })
elif RECEIVED_PARAMS['dataset'] == 'amigos' and RECEIVED_PARAMS[
        'target'] == 'valence':
    RECEIVED_PARAMS.update({
        "generator_lr": 0.0001,
        "discriminator_lr": 0.0001,
        "weight_decay": 0,
        "w_boundary": 5,
        "w_smooth": 0.85,
        "num_epochs": 100,
        "dataset": "amigos",
        "target": "valence"
    })
elif RECEIVED_PARAMS['dataset'] == 'amigos' and RECEIVED_PARAMS[
        'target'] == 'arousal':
    RECEIVED_PARAMS.update({
        "generator_lr": 0.0001,
        "discriminator_lr": 0.0001,
        "weight_decay": 0,
        "w_boundary": 5,
        "w_smooth": 0.85,
        "num_epochs": 100,
        "dataset": "amigos",
        "target": "arousal"
    })
elif RECEIVED_PARAMS['dataset'] == 'seed' and RECEIVED_PARAMS[
    'target'] == 'emotion':
    RECEIVED_PARAMS.update({
        "generator_lr": 0.0001,
        "discriminator_lr": 0.0001,
        "weight_decay": 0,
        "w_diverse": 1,
        "w_boundary": 5,
        "w_smooth": 0.85,
        "num_epochs": 100,
        "dataset": "seed",
        "target": "emotion"
    })
else:
    raise NotImplementedError

if RECEIVED_PARAMS['dataset'] == 'deap':
    RECEIVED_PARAMS = {
        **RECEIVED_PARAMS, 'sampling_rate': 128,
        'chunk_size': 128,
        'baseline_chunk_size': 128,
        'num_baseline': 3,
        'channel_list': DEAP_CHANNEL_LIST,
        'channel_location_dict': DEAP_CHANNEL_LOCATION_DICT,
        'io_path': './tmp_out/deap',
        'root_path': './tmp_in/data_preprocessed_python',
        'split_path': './tmp_out/5_fold_group_by_trial/split',
        'tensorboard_path': './tmp_out/train_gan/vis',
        'logging_path': './tmp_out/train_gan/log',
        'gan_state_dict_path': './tmp_out/train_gan/weight/',
        'evaluator_state_dict_path': './tmp_out/train_evaluator/weight/'
    }
elif RECEIVED_PARAMS['dataset'] == 'amigos':
    RECEIVED_PARAMS = {
        **RECEIVED_PARAMS, 'sampling_rate': 128,
        'chunk_size': 128,
        'baseline_chunk_size': 128,
        'num_baseline': 5,
        'channel_list': AMIGOS_CHANNEL_LIST,
        'channel_location_dict': AMIGOS_CHANNEL_LOCATION_DICT,
        'io_path': './tmp_out/amigos',
        'root_path': './tmp_in/data_preprocessed',
        'split_path': './tmp_out/5_fold_group_by_trial_amigos/split',
        'tensorboard_path': './tmp_out/train_gan_amigos/vis',
        'logging_path': './tmp_out/train_gan_amigos/log',
        'gan_state_dict_path': './tmp_out/train_gan_amigos/weight/',
        'evaluator_state_dict_path': './tmp_out/train_evaluator_amigos/weight/'
    }
elif RECEIVED_PARAMS['dataset'] == 'seed':
    RECEIVED_PARAMS = {
        **RECEIVED_PARAMS, 'channel_list': SEED_CHANNEL_LIST,
        'channel_location_dict': SEED_CHANNEL_LOCATION_DICT,
        'io_path': './tmp_out/seed',
        'root_path': './tmp_in/ExtractedFeatures',
        'split_path': './tmp_out/5_fold_group_by_trial_seed/split',
        'tensorboard_path': './tmp_out/train_gan_seed/vis',
        'logging_path': './tmp_out/train_gan_seed/log',
        'gan_state_dict_path': './tmp_out/train_gan_seed/weight/',
        'evaluator_state_dict_path': './tmp_out/train_evaluator_seed/weight/',
        'target': 'emotion',
        'num_classes': 3,
        'num_bands': 5
    }
else:
    raise NotImplementedError

timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
RECEIVED_PARAMS['tensorboard_path'] = os.path.join(
    RECEIVED_PARAMS['tensorboard_path'], f'{TRAIL_ID}_{timeticks}')
RECEIVED_PARAMS['gan_state_dict_path'] = os.path.join(
    RECEIVED_PARAMS['gan_state_dict_path'], f'{TRAIL_ID}_{timeticks}')

os.makedirs(RECEIVED_PARAMS['logging_path'], exist_ok=True)
os.makedirs(RECEIVED_PARAMS['gan_state_dict_path'], exist_ok=True)

logger = logging.getLogger('train the gan and record the GAN-related metrics')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    os.path.join(RECEIVED_PARAMS['logging_path'],
                 f'{TRAIL_ID}_{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(RECEIVED_PARAMS)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


writer = SummaryWriter(log_dir=RECEIVED_PARAMS['tensorboard_path'],
                       comment='Train GAN')


class MyFrechetInceptionDistance(FrechetInceptionDistance):
    def __init__(
        self,
        feature,
        reset_real_features=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inception = feature
        dummy_image = torch.randn(2, RECEIVED_PARAMS['num_bands'], 9, 9)
        num_features = self.inception(dummy_image).shape[-1]

        if not isinstance(reset_real_features, bool):
            raise ValueError(
                "Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        mx_nb_feets = (num_features, num_features)
        self.add_state("real_features_sum",
                       torch.zeros(num_features).double(),
                       dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum",
                       torch.zeros(mx_nb_feets).double(),
                       dist_reduce_fx="sum")
        self.add_state("real_features_num_samples",
                       torch.tensor(0).long(),
                       dist_reduce_fx="sum")

        self.add_state("fake_features_sum",
                       torch.zeros(num_features).double(),
                       dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum",
                       torch.zeros(mx_nb_feets).double(),
                       dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples",
                       torch.tensor(0).long(),
                       dist_reduce_fx="sum")

    def update(self, imgs, real):  # type: ignore
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]


class MyInceptionScore(InceptionScore):
    def update(self, imgs):
        features = self.inception(imgs)
        self.features.append(features)


class MyKernelInceptionDistance(KernelInceptionDistance):
    def update(self, imgs, real):  # type: ignore
        features = self.inception(imgs)
        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)


class ESCGANTrainer(CGANTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)

    def on_training_step(self, train_batch, batch_id, num_batches):
        self.train_g_loss.reset()
        self.train_d_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        valid = torch.zeros((X.shape[0], 1),
                            requires_grad=False).to(self.device)
        fake = torch.ones((X.shape[0], 1), requires_grad=False).to(self.device)

        self.generator_optimizer.zero_grad()

        aux_loss_fn = nn.CrossEntropyLoss()

        rec_X, _ = self.modules['generator'](X, y)
        g_loss = F.mse_loss(X, rec_X)

        gen_y = torch.abs(1 - y).long()
        gen_X, gen_Z = self.modules['generator'](X, gen_y)
        pred_valid_fake, pred_gen_y = self.modules['discriminator'](gen_X)
        g_loss += 0.5 * self.loss_fn(pred_valid_fake, valid)

        boundary_loss = 0.5 * F.kl_div(
                F.log_softmax(pred_gen_y, dim=1),
                torch.ones_like(pred_gen_y) / pred_gen_y.shape[1],
                reduction='batchmean')

        g_loss += RECEIVED_PARAMS['w_boundary'] * boundary_loss

        indices = torch.randperm(X.shape[0])
        perm_gen_X = gen_X[indices]
        perm_gen_Z = gen_Z[indices]

        diverse_loss = torch.exp(
            -torch.abs(gen_X - perm_gen_X).mean([-1, -2, -3]) /
            (1e-6 + torch.abs(gen_Z - perm_gen_Z).mean([-1, -2, -3]))).mean()

        g_loss += RECEIVED_PARAMS['w_diverse'] * diverse_loss

        g_loss.backward()
        self.generator_optimizer.step()

        self.discriminator_optimizer.zero_grad()

        pred_valid_fake, pred_y = self.modules['discriminator'](X)

        if batch_id % 5 == 0:
            real_loss = 0.5 * self.loss_fn(
                pred_valid_fake, valid) + 0.5 * aux_loss_fn(pred_y, y)

            pred_valid_fake, pred_gen_y = self.modules['discriminator'](
                gen_X.detach())
            fake_loss = 0.5 * self.loss_fn(
                pred_valid_fake, fake * RECEIVED_PARAMS['w_smooth']
            ) + 0.5 * aux_loss_fn(pred_gen_y, gen_y)
            d_loss = 0.5 * real_loss + 0.5 * fake_loss
        else:
            d_loss = aux_loss_fn(pred_y, y)

        d_loss.backward()
        self.discriminator_optimizer.step()

        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_g_loss.update(g_loss)
            self.train_d_loss.update(d_loss)

            train_g_loss = self.train_g_loss.compute()
            train_d_loss = self.train_d_loss.compute()

            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"g_loss: {train_g_loss:>8f}, d_loss: {train_d_loss:>8f} [{batch_id:>5d}/{num_batches:>5d}]"
                )

    def before_test_epoch(self, extractor=None, predictor=None, **kwargs):
        assert not extractor is None, 'A pretrained feature extraction model needs to be provided to compute the kernel MMD, IS and FID scores!'
        
        self.test_kernel_mmd = MyKernelInceptionDistance(
            feature=extractor.cpu()).to(self.device)
        self.test_is = MyInceptionScore(feature=predictor.cpu()).to(self.device)
        self.test_fid = MyFrechetInceptionDistance(feature=extractor.cpu()).to(
            self.device)

    def on_test_step(self, test_batch, batch_id, num_batches, **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        gen_y = torch.randint(0, self.modules['discriminator'].num_classes,
                              (X.shape[0], )).to(self.device)
        gen_X, _ = self.modules['generator'](X, gen_y)

        self.test_kernel_mmd.update(X, real=True)
        self.test_kernel_mmd.update(gen_X, real=False)

        self.test_is.update(gen_X)
        self.test_fid.update(X, real=True)
        self.test_fid.update(gen_X, real=False)

        vis_batch = num_batches // 5
        if batch_id % vis_batch == 0:
            t = transforms.ToInterpolatedGrid(
                RECEIVED_PARAMS['channel_location_dict'])
            signal = t.reverse(eeg=gen_X[0].detach().cpu().numpy())['eeg']
            top_img = plot_feature_topomap(
                torch.tensor(signal),
                channel_list=RECEIVED_PARAMS['channel_list'],
                feature_list=["delta", "theta", "alpha", "beta", "gamma"])
            writer.add_image(f'top{batch_id}/eeg-gen-{gen_y[0]}',
                             top_img,
                             self.cur_epoch,
                             dataformats='HWC')

            signal = t.reverse(eeg=X[0].detach().cpu().numpy())['eeg']
            top_img = plot_feature_topomap(
                torch.tensor(signal),
                channel_list=RECEIVED_PARAMS['channel_list'],
                feature_list=["delta", "theta", "alpha", "beta", "gamma"])
            writer.add_image(f'top{batch_id}/eeg-gt-{y[0]}',
                             top_img,
                             self.cur_epoch,
                             dataformats='HWC')

    def after_test_epoch(self, **kwargs):
        test_kernel_mmd = self.test_kernel_mmd.compute()[0].item()
        test_is = self.test_is.compute()[0].item()
        test_fid = self.test_fid.compute().item()

        return test_kernel_mmd, test_is, test_fid

    def test(self, test_loader, **kwargs):
        test_loader = self.on_reveive_dataloader(test_loader, mode='test')

        for k, m in self.modules.items():
            self.modules[k].eval()

        num_batches = len(test_loader)
        self.before_test_epoch(**kwargs)
        for batch_id, test_batch in enumerate(test_loader):
            self.before_test_step(batch_id, num_batches, **kwargs)
            self.on_test_step(test_batch, batch_id, num_batches, **kwargs)
            self.after_test_step(batch_id, num_batches, **kwargs)
        return self.after_test_epoch(**kwargs)

    def before_validation_epoch(self, epoch_id, num_epochs, **kwargs):
        self.cur_epoch = epoch_id
        self.val_a_accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=self.num_classes,
            top_k=1).to(self.device)
        self.val_a_g_accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=self.num_classes,
            top_k=1).to(self.device)

    def on_validation_step(self, val_batch, batch_id, num_batches, **kwargs):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        gen_y = torch.abs(1 - y).long()
        gen_X, _ = self.modules['generator'](X, gen_y)

        _, pred_y = self.modules['discriminator'](X)
        _, pred_gen_y = self.modules['discriminator'](gen_X.detach())

        self.val_a_accuracy.update(pred_y.argmax(1), y)
        self.val_a_g_accuracy.update(pred_gen_y.argmax(1), gen_y)

    def after_validation_epoch(self, epoch_id, num_epochs):
        val_a_accuracy = 100 * self.val_a_accuracy.compute()
        val_a_g_accuracy = 100 * self.val_a_g_accuracy.compute()

        self.log(
            f"\n d accuracy (real): {val_a_accuracy:>8f}%\n d accuracy (fake): {val_a_g_accuracy:>8f}%"
        )

        val_a_accuracy = val_a_accuracy.item()
        val_a_g_accuracy = val_a_g_accuracy.item()

        nni.report_intermediate_result({
            'default': val_a_accuracy,
            'val_a_accuracy': val_a_accuracy,
            'val_a_g_accuracy': val_a_g_accuracy
        })


if __name__ == "__main__":
    if RECEIVED_PARAMS['num_bands'] == 4:
        BAND_DICT = {
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 45]
        }
    elif RECEIVED_PARAMS['num_bands'] == 5:
        BAND_DICT = {
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 45]
        }
    if __name__ == "__main__":
        if RECEIVED_PARAMS['num_bands'] == 4:
            BAND_DICT = {
                "theta": [4, 8],
                "alpha": [8, 13],
                "beta": [13, 30],
                "gamma": [30, 45]
            }
        elif RECEIVED_PARAMS['num_bands'] == 5:
            BAND_DICT = {
                "delta": [1, 4],
                "theta": [4, 8],
                "alpha": [8, 13],
                "beta": [13, 30],
                "gamma": [30, 45]
            }

    if RECEIVED_PARAMS['dataset'] == 'deap':
        dataset = DEAPDataset(
            io_path=RECEIVED_PARAMS['io_path'],
            root_path=RECEIVED_PARAMS['root_path'],
            offline_transform=transforms.Compose([
                transforms.BandDifferentialEntropy(band_dict=BAND_DICT,
                                                   apply_to_baseline=True),
                transforms.BaselineRemoval(),
                transforms.ToInterpolatedGrid(
                    RECEIVED_PARAMS['channel_location_dict'])
            ]),
            online_transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            label_transform=transforms.Compose([
                transforms.Select(RECEIVED_PARAMS['target']),
                transforms.Binary(5.0),
            ]),
            chunk_size=RECEIVED_PARAMS['chunk_size'],
            baseline_chunk_size=RECEIVED_PARAMS['baseline_chunk_size'],
            num_baseline=RECEIVED_PARAMS['num_baseline'],
            num_worker=4,
            in_memory=False)
    elif RECEIVED_PARAMS['dataset'] == 'amigos':
        dataset = AMIGOSDataset(
            io_path=RECEIVED_PARAMS['io_path'],
            root_path=RECEIVED_PARAMS['root_path'],
            offline_transform=transforms.Compose([
                transforms.BandDifferentialEntropy(band_dict=BAND_DICT,
                                                   apply_to_baseline=True),
                transforms.BaselineRemoval(),
                transforms.ToInterpolatedGrid(
                    RECEIVED_PARAMS['channel_location_dict'])
            ]),
            online_transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            label_transform=transforms.Compose([
                transforms.Select(RECEIVED_PARAMS['target']),
                transforms.Binary(5.0)
            ]),
            chunk_size=RECEIVED_PARAMS['chunk_size'],
            baseline_chunk_size=RECEIVED_PARAMS['baseline_chunk_size'],
            num_baseline=RECEIVED_PARAMS['num_baseline'],
            num_worker=4,
            in_memory=False)
    elif RECEIVED_PARAMS['dataset'] == 'seed':
        dataset = SEEDFeatureDataset(
            io_path=RECEIVED_PARAMS['io_path'],
            root_path=RECEIVED_PARAMS['root_path'],
            feature=['de_movingAve'],
            offline_transform=transforms.Compose([
                transforms.MinMaxNormalize(axis=-1),
                transforms.ToInterpolatedGrid(
                    RECEIVED_PARAMS['channel_location_dict'])
            ]),
            online_transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            label_transform=transforms.Compose([
                transforms.Select(RECEIVED_PARAMS['target']),
                transforms.Lambda(lambda x: int(x) + 1),
            ]),
            num_worker=4,
            in_memory=False)
    else:
        raise NotImplementedError

    k_fold = KFoldGroupbyTrial(n_splits=5,
                               split_path=RECEIVED_PARAMS['split_path'])

    kernel_mmd_scores = []
    is_scores = []
    fid_scores = []

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        generator = ESCGenerator(
            hid_channels=RECEIVED_PARAMS['hid_channels'],
            out_channels=RECEIVED_PARAMS['num_bands'],
            num_classes=RECEIVED_PARAMS['num_classes'],
            channel_location_dict=DEAP_CHANNEL_LOCATION_DICT,
            with_bn=True)
        discriminator = ESCDiscriminator(
            hid_channels=RECEIVED_PARAMS['hid_channels'],
            in_channels=RECEIVED_PARAMS['num_bands'],
            num_classes=RECEIVED_PARAMS['num_classes'],
            with_bn=True)

        trainer = ESCGANTrainer(
            generator=generator,
            discriminator=discriminator,
            generator_lr=RECEIVED_PARAMS['generator_lr'],
            discriminator_lr=RECEIVED_PARAMS['discriminator_lr'],
            weight_decay=RECEIVED_PARAMS['weight_decay'],
            device_ids=[0])

        train_loader = DataLoader(train_dataset,
                                  batch_size=RECEIVED_PARAMS['batch_size'],
                                  shuffle=True,
                                  num_workers=4)
        val_loader = DataLoader(val_dataset,
                                batch_size=RECEIVED_PARAMS['batch_size'],
                                shuffle=True,
                                num_workers=4)

        trainer.fit(train_loader,
                    val_loader,
                    num_epochs=RECEIVED_PARAMS['num_epochs'])

        extractor = Extractor(in_channels=5, hid_channels=128, num_classes=2)
        extractor.load_state_dict(
            torch.load(
                os.path.join(RECEIVED_PARAMS['evaluator_state_dict_path'],
                             f'{i}.pth'))['model'])
        extractor.eval()

        predictor = Predictor(in_channels=5, hid_channels=128, num_classes=2)
        predictor.load_state_dict(
            torch.load(
                os.path.join(RECEIVED_PARAMS['evaluator_state_dict_path'],
                             f'{i}.pth'))['model'])
        predictor.eval()

        kernel_mmd_score, is_score, fid_score = trainer.test(
            val_loader, extractor=extractor, predictor=predictor)

        kernel_mmd_scores.append(kernel_mmd_score)
        is_scores.append(is_score)
        fid_scores.append(fid_score)

        trainer.save_state_dict(
            os.path.join(RECEIVED_PARAMS['gan_state_dict_path'], f'{i}.pth'))

    nni.report_final_result({
        'default': np.array(kernel_mmd_scores).mean(),
        'kernel_mmd': np.array(kernel_mmd_scores).mean(),
        'is': np.array(is_scores).mean(),
        'fid': np.array(fid_scores).mean()
    })
