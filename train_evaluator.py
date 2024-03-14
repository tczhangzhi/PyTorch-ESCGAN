import logging
import math
import os
import random
import time

import nni
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data.dataloader import DataLoader
from torchattacks import PGD, DeepFool

from models import ESCClassifier
from torcheeg import transforms
from torcheeg.datasets import (AMIGOSDataset, DEAPDataset, DREAMERDataset,
                               SEEDFeatureDataset)
from torcheeg.datasets.constants.emotion_recognition.amigos import (
    AMIGOS_CHANNEL_LIST, AMIGOS_CHANNEL_LOCATION_DICT)
from torcheeg.datasets.constants.emotion_recognition.deap import (
    DEAP_CHANNEL_LIST, DEAP_CHANNEL_LOCATION_DICT)
from torcheeg.datasets.constants.emotion_recognition.seed import (
    SEED_CHANNEL_LIST, SEED_CHANNEL_LOCATION_DICT)
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.trainers import ClassificationTrainer

TRAIL_ID = nni.get_trial_id()

RECEIVED_PARAMS = {
    'hid_channels': 128,
    'num_classes': 3,
    'lr': 0.0001,
    'weight_decay': 0.0001,
    'batch_size': 256,
    'n_splits': 5,
    'num_bands': 5,
    'num_epochs': 100,
    'target': 'emotion',
    'dataset': 'seed'
}

if not TRAIL_ID == 'STANDALONE':
    RECEIVED_PARAMS.update(nni.get_next_parameter())

if RECEIVED_PARAMS['dataset'] == 'deap':
    
    RECEIVED_PARAMS = {
        **RECEIVED_PARAMS,
        'sampling_rate':
        128,
        'chunk_size':
        128,
        'baseline_chunk_size':
        128,
        'num_baseline':
        3,
        'channel_list':
        DEAP_CHANNEL_LIST,
        'channel_location_dict':
        DEAP_CHANNEL_LOCATION_DICT,
        'io_path':
        './tmp_out/deap',
        'root_path':
        './tmp_in/data_preprocessed_python',
        'split_path':
        './tmp_out/5_fold_group_by_trial/split',
        'logging_path':
        './tmp_out/train_evaluator/log',
        'evaluator_state_dict_path':
        './tmp_out/train_evaluator/weight/'
    }

elif RECEIVED_PARAMS['dataset'] == 'amigos':

    RECEIVED_PARAMS = {
        **RECEIVED_PARAMS,
        'sampling_rate':
        128,
        'chunk_size':
        128,
        'baseline_chunk_size':
        128,
        'num_baseline':
        5,
        'channel_list':
        AMIGOS_CHANNEL_LIST,
        'channel_location_dict':
        AMIGOS_CHANNEL_LOCATION_DICT,
        'io_path':
        './tmp_out/amigos',
        'root_path':
        './tmp_in/data_preprocessed',
        'split_path':
        './tmp_out/5_fold_group_by_trial_amigos/split',
        'logging_path':
        './tmp_out/train_evaluator_amigos/log',
        'evaluator_state_dict_path':
        './tmp_out/train_evaluator_amigos/weight/'
    }

elif RECEIVED_PARAMS['dataset'] == 'seed':
    
    RECEIVED_PARAMS = {
        **RECEIVED_PARAMS, 'num_classes': 3,
        'target': 'emotion',
        'channel_list': SEED_CHANNEL_LIST,
        'channel_location_dict': SEED_CHANNEL_LOCATION_DICT,
        'io_path': './tmp_out/seed',
        'root_path': './tmp_in/ExtractedFeatures',
        'split_path': './tmp_out/5_fold_group_by_trial_seed/split',
        'logging_path': './tmp_out/train_evaluator_seed/log',
        'evaluator_state_dict_path': './tmp_out/train_evaluator_seed/weight/'
    }
else:
    raise NotImplementedError

timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
os.makedirs(RECEIVED_PARAMS['logging_path'], exist_ok=True)

logger = logging.getLogger(
    'train evaluator (baseline and the evaluator for GAN features)')
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


def fgsm_attack(eeg, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_eeg = eeg + epsilon * sign_data_grad
    return perturbed_eeg


def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get('_attacks').values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result

    return wrapper_func


class EEGPGD(PGD):
    @wrapper_method
    def _check_inputs(self, images):
        pass

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = adv_images.detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost,
                                       adv_images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps,
                                max=self.eps)
            adv_images = images + delta

        return adv_images


class EEGDeepFool(DeepFool):
    @wrapper_method
    def _check_inputs(self, images):
        pass

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.get_logits(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))

        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (1 + self.overshoot) * delta
        adv_image = adv_image.detach()
        return (False, target_label, adv_image)


class ESCClassifierTrainer(ClassificationTrainer):
    def __init__(self,
                 model,
                 num_classes=None,
                 temperature=4.0,
                 lr=1e-4,
                 weight_decay=0.0,
                 device_ids=[],
                 ddp_sync_bn=True,
                 ddp_replace_sampler=True,
                 ddp_val=True,
                 ddp_test=True):
        super(ClassificationTrainer,
              self).__init__(modules={'model': model},
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature

        if not num_classes is None:
            self.num_classes = num_classes
        elif hasattr(model, 'num_classes'):
            self.num_classes = model.num_classes
        else:
            raise ValueError('The number of classes is not specified.')

        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_loss = torchmetrics.MeanMetric().to(self.device)
        self.train_accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=self.num_classes,
            top_k=1).to(self.device)

        self.val_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass',
                                                  num_classes=self.num_classes,
                                                  top_k=1).to(self.device)

        self.test_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass',
                                                   num_classes=self.num_classes,
                                                   top_k=1).to(self.device)

    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)

    def after_validation_epoch(self, epoch_id, num_epochs, **kwargs):
        val_accuracy = 100 * self.val_accuracy.compute()
        val_loss = self.val_loss.compute()
        nni.report_intermediate_result({
            'default': val_accuracy.item(),
            'val_loss': val_loss.item(),
            'val_accuracy': val_accuracy.item()
        })
        self.log(f"\nloss: {val_loss:>8f}, accuracy: {val_accuracy:>0.1f}%")

    def after_test_epoch(self, **kwargs):
        test_accuracy = 100 * self.test_accuracy.compute()
        test_loss = self.test_loss.compute()

        test_attack_accuracy = 100 * self.test_attack_accuracy.compute()
        test_attack_loss = self.test_attack_loss.compute()

        self.log(
            f"\nloss: {test_loss:>8f}, accuracy: {test_accuracy:>8f}%\nloss after attack: {test_attack_loss:>8f}, accuracy after attack: {test_attack_accuracy:>8f}%"
        )

        return test_accuracy.item(), test_attack_accuracy.item()

    def before_test_epoch(self, **kwargs):
        self.test_loss.reset()
        self.test_accuracy.reset()

        self.test_attack_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_attack_accuracy = torchmetrics.Accuracy(
            task='multiclass', num_classes=self.num_classes,
            top_k=1).to(self.device)

    def on_test_step(self,
                     test_batch,
                     batch_id,
                     num_batches,
                     epsilon=0.01,
                     **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        X.requires_grad = True

        pred = self.modules['model'](X)

        self.test_loss.update(self.loss_fn(pred, y))
        self.test_accuracy.update(pred.argmax(1), y)

        attacker = EEGPGD(self.modules['model'],
                          eps=epsilon,
                          alpha=0.001,
                          steps=10,
                          random_start=False)
        perturbed_X = attacker(X, y)

        pred = self.modules['model'](perturbed_X)

        self.test_attack_loss.update(self.loss_fn(pred, y))
        self.test_attack_accuracy.update(pred.argmax(1), y)

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

    def on_training_step(self, train_batch, batch_id, num_batches, **kwargs):
        self.train_accuracy.reset()
        self.train_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        pred = self.modules['model'](X)
        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_loss.update(loss)
            self.train_accuracy.update(pred.argmax(1), y)

            train_loss = self.train_loss.compute()
            train_accuracy = 100 * self.train_accuracy.compute()

            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"loss: {train_loss:>8f}, accuracy: {train_accuracy:>8f}% [{batch_id:>5d}/{num_batches:>5d}]"
                )


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
            offline_transform=transforms.Compose([
                transforms.MinMaxNormalize(axis=-1),
                transforms.ToInterpolatedGrid(
                    RECEIVED_PARAMS['channel_location_dict'])
            ]),
            online_transform=transforms.ToTensor(
            ),
            label_transform=transforms.Compose([
                transforms.Select(RECEIVED_PARAMS['target']),
                transforms.Lambda(lambda x: x + 1)
            ]),
            feature=['de_LDS'],
            num_worker=4,
            in_memory=False)

    elif RECEIVED_PARAMS['dataset'] == 'dreamer':
        dataset = DREAMERDataset(
            io_path=RECEIVED_PARAMS['io_path'],
            mat_path=RECEIVED_PARAMS['mat_path'],
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
                transforms.Binary(3.0)
            ]),
            chunk_size=RECEIVED_PARAMS['chunk_size'],
            baseline_chunk_size=RECEIVED_PARAMS['baseline_chunk_size'],
            num_baseline=RECEIVED_PARAMS['num_baseline'],
            num_worker=4,
            in_memory=False)
    else:
        raise NotImplementedError

    k_fold = KFoldGroupbyTrial(n_splits=5,
                               split_path=RECEIVED_PARAMS['split_path'])

    test_accuracies = []
    test_attack_accuracies = []

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):

        train_loader = DataLoader(
            train_dataset,
            batch_size=RECEIVED_PARAMS['batch_size'],
            shuffle=True,
            num_workers=4)
        val_loader = DataLoader(
            val_dataset,
            batch_size=RECEIVED_PARAMS['batch_size'],
            shuffle=True,
            num_workers=4)

        classifier = ESCClassifier(hid_channels=RECEIVED_PARAMS['hid_channels'],
                                 in_channels=RECEIVED_PARAMS['num_bands'],
                                 num_classes=RECEIVED_PARAMS['num_classes'],
                                 with_bn=True)

        cls_trainer = ESCClassifierTrainer(
            model=classifier,
            lr=RECEIVED_PARAMS['lr'],
            weight_decay=RECEIVED_PARAMS['weight_decay'],
            device_ids=[0])

        cls_trainer.fit(train_loader,
                            val_loader,
                            num_epochs=RECEIVED_PARAMS['num_epochs'])

        test_accuracy, test_attack_accuracy = cls_trainer.test(val_loader,
                                                               epsilon=0.01)

        test_accuracies.append(test_accuracy)
        test_attack_accuracies.append(test_attack_accuracy)

        cls_trainer.save_state_dict(
                os.path.join(RECEIVED_PARAMS['evaluator_state_dict_path'],
                             f'{i}.pth'))

    nni.report_final_result({
        'default':
        np.array(test_accuracies).mean(),
        'test_accuracy':
        np.array(test_accuracies).mean(),
        'attack_accuracy':
        np.array(test_attack_accuracies).mean(),
    })