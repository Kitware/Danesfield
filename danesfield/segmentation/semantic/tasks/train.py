###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import os
from collections import defaultdict
from osgeo import gdal
import gc

import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm
from typing import Type

from dataset.neural_dataset import TrainDataset, ValDataset
from .loss import dice_round, dice, BCEDiceLoss, BCELoss, BCEDiceLossWeighted, BCELossWeighted
from .callbacks import ModelSaver, TensorBoard, CheckpointSaver, Callbacks, StepLR, InverseLR
from models import extension_unet
from models import resnet_unet
from models import dense_unet
import numpy as np

torch.backends.cudnn.benchmark = True

models = {
    'extensionunet': extension_unet.ExtensionUNet,
    'resnet34': resnet_unet.ResnetUNet,
    'denseunet': dense_unet.DenseUNet
}

losses = {
    'bce_loss': BCELoss,
    'bce_loss_w': BCELossWeighted,
    'bce_dice_loss': BCEDiceLoss,
    'bce_dice_loss_w': BCEDiceLossWeighted
}

optimizers = {
    'adam': optim.Adam
}


class Estimator:
    """
    class takes care about model, optimizer, loss and making step in parameter space
    """
    def __init__(self, model: torch.nn.Module, optimizer: Type[optim.Optimizer],
                 loss: Type[torch.nn.Module], save_path,
                 iter_size=1, test_iter_size=1, lr=1e-4, num_channels_changed=False):
        self.model = nn.DataParallel(model).cuda()
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.criterion = loss().cuda()
        self.iter_size = iter_size
        self.test_iter_size = test_iter_size
        self.start_epoch = 0
        self.start_step = 0
        # self.best_val_bce = 1.0e+6
        # self.best_val_dice = -1.0e+6
        self.best_val_bce = {'D1': 1.0e+6, 'D2': 1.0e+6, 'D3': 1.0+6, 'D4': 1.0e+6}
        self.best_val_dice = {'D1': -1.0e+6, 'D2': -1.0e+6, 'D3': -1.0e+6, 'D4': -1.0e+6}
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.num_channels_changed = num_channels_changed

        self.lr_scheduler = None
        self.lr = lr
        self.optimizer_type = optimizer

    def resume(self, checkpoint_name):
        try:
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint_name))
        except FileNotFoundError:
            print("resume failed, file not found")
            return False

        self.start_epoch = checkpoint['epoch']
        self.start_step = checkpoint['step']
        self.best_val_bce = checkpoint['best_val_bce']
        self.best_val_dice = checkpoint['best_val_dice']

        model_dict = self.model.module.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if self.num_channels_changed:
            skip_layers = self.model.module.first_layer_params_names
            print('skipping: ', [k for k in pretrained_dict.keys()
                  if any(s in k for s in skip_layers)])
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if not any(s in k for s in skip_layers)}
            model_dict.update(pretrained_dict)
            self.model.module.load_state_dict(model_dict)
        else:
            model_dict.update(pretrained_dict)
            try:
                self.model.module.load_state_dict(model_dict)
            except Exception:
                print('load state dict failed')
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception:
                pass

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        print("resumed from checkpoint {} on epoch={}/step={}".format(os.path.join(self.save_path,
              checkpoint_name), self.start_epoch, self.start_step))

        return True

    def index_in_copy_out(self, width):
        if width < 1024:
            print('Width is smaller than 1024. This is not enough to split')
            return None

        nwids = int(np.ceil(width/1024.0))
        cpwid = int(width/nwids)

        if cpwid % 2 == 1:
            cpwid += 1

        insidx = [0]
        ineidx = [1024]
        cpsidx = [0]
        cpeidx = [cpwid]
        outsidx = [0]
        outeidx = [cpwid]

        for i in range(1, nwids-1, 1):
            ins = int((i+0.5)*cpwid - 512)
            ine = int(ins + 1024)
            cps = int(512-cpwid*0.5)
            cpe = int(512+cpwid*0.5)
            outs = int(cpwid*i)
            oute = int(cpwid*(i+1))

            insidx.append(ins)
            ineidx.append(ine)
            cpsidx.append(cps)
            cpeidx.append(cpe)
            outsidx.append(outs)
            outeidx.append(oute)

        rest = width - (nwids-1)*cpwid
        insidx.append(width-1024)
        ineidx.append(width)
        cpsidx.append(1024-rest)
        cpeidx.append(1024)
        outsidx.append(width-rest)
        outeidx.append(width)

        return insidx, ineidx, cpsidx, cpeidx, outsidx, outeidx

    def make_step_itersize_large(self, images, ytrues, training, metrics):
        iter_size = self.test_iter_size
        if training:
            self.optimizer.zero_grad()

        batch_iter_size = images.size()[0]/iter_size

        inputs = images.chunk(iter_size)
        targets = ytrues.chunk(iter_size)
        outputs = []

        meter = {k: 0 for k, v in metrics}
        meter['loss'] = 0
        for input, target in zip(inputs, targets):
            dsize = input.size()
            # print('input.size(): {}'.format(input.size()))
            target = torch.autograd.Variable(target.cuda(async=True), volatile=not training)

            xinsidx, xineidx, xcpsidx, xcpeidx, xoutsidx, xouteidx = self.index_in_copy_out(
                    dsize[2])
            yinsidx, yineidx, ycpsidx, ycpeidx, youtsidx, youteidx = self.index_in_copy_out(
                    dsize[3])

            output = torch.autograd.Variable(torch.zeros((dsize[0], 1, dsize[2], dsize[3])).cuda())
            for i in range(len(xinsidx)):
                for j in range(len(yinsidx)):
                    samples = torch.autograd.Variable(input[:, :, xinsidx[i]:xineidx[i],
                                                      yinsidx[j]:yineidx[j]],
                                                      volatile=not training).cuda()
                    prediction = self.model(samples)
                    output[:, :, xoutsidx[i]:xouteidx[i], youtsidx[j]:youteidx[j]
                           ] = prediction[:, :, xcpsidx[i]:xcpeidx[i], ycpsidx[j]:ycpeidx[j]]

            # print('output: {}'.format(output))
            # print('target: {}'.format(target))

            reg_lambda = 0.001
            reg_loss = 0
            for param in self.model.parameters():
                    reg_loss += param.norm(2)

            loss = self.criterion(output, target) / batch_iter_size + reg_lambda*reg_loss
            # loss /= batch_size

            if training:
                loss.backward()

            meter['loss'] += loss.data.cpu().numpy()[0]
            output = torch.sigmoid(output)
            for name, func in metrics:
                acc = func(output.contiguous(), target.contiguous())
                meter[name] += acc.data.cpu().numpy()[0] / batch_iter_size

            outputs.append(output.data)

        if training:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
            self.optimizer.step()
        return meter, torch.cat(outputs, dim=0)

    def make_step_itersize(self, images, ytrues, training, metrics):
        iter_size = self.iter_size
        if training:
            self.optimizer.zero_grad()

        batch_iter_size = images.size()[0]/iter_size

        inputs = images.chunk(iter_size)
        targets = ytrues.chunk(iter_size)
        outputs = []

        meter = {k: 0 for k, v in metrics}
        meter['loss'] = 0
        for input, target in zip(inputs, targets):
            # print('input.size(): {}'.format(input.size()))
            input = torch.autograd.Variable(input.cuda(async=True), volatile=not training)
            target = torch.autograd.Variable(target.cuda(async=True), volatile=not training)
            output = self.model(input)

            reg_lambda = 0.001
            reg_loss = 0
            for param in self.model.parameters():
                    reg_loss += param.norm(2)

            loss = self.criterion(output, target) / batch_iter_size + reg_lambda*reg_loss

            if training:
                loss.backward()

            meter['loss'] += loss.data.cpu().numpy()[0]
            output = torch.sigmoid(output)
            for name, func in metrics:
                acc = func(output.contiguous(), target.contiguous())
                # print('size = {}, {} = {}'.format(target.size(), name, acc.data.cpu().numpy()[0]))
                meter[name] += acc.data.cpu().numpy()[0] / batch_iter_size

            outputs.append(output.data)

        if training:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
            self.optimizer.step()
        return meter, torch.cat(outputs, dim=0)


class MetricsCollection:
    def __init__(self):
        self.stop_training = False
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class PytorchTrain:
    """
    class for training process and callbacks in right places
    """
    def __init__(self, estimator: Estimator, fold, metrics, callbacks=None,
                 hard_negative_miner=None):
        self.fold = fold
        self.step = 0
        self.save_every_steps = 20
        self.estimator = estimator
        self.metrics = metrics
        self.best_val_bce = {'D1': 1.0e+6, 'D2': 1.0e+6, 'D3': 1.0+6, 'D4': 1.0e+6}
        self.best_val_dice = {'D1': -1.0e+6, 'D2': -1.0e+6, 'D3': -1.0e+6, 'D4': -1.0e+6}

        # self.devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        if os.name == 'nt':
            self.devices = ','.join(str(d + 5) for d in map(int, self.devices.split(',')))

        self.hard_negative_miner = hard_negative_miner
        self.metrics_collection = MetricsCollection()

        # use multi-fold
        if fold >= 0:
            self.estimator.resume("fold" + str(fold) + "_checkpoint.pth")
        else:  # use onetrain
            self.estimator.resume("onetrain_checkpoint.pth")

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def _run_training(self, epoch, train_loader, val_loader, foldername):
        avg_meter = defaultdict(float)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc="Epoch={}/train".format(epoch), ncols=0)
        for i, data in pbar:
            self.callbacks.on_batch_begin(i)
            self.step += 1
            self.callbacks.on_step_begin(epoch, self.step)

            meter, ypreds = self._make_step(data, True)
            for k, val in meter.items():
                avg_meter[k] += val

            if self.hard_negative_miner is not None:
                self.hard_negative_miner.update_cache(meter, data)
                if self.hard_negative_miner.need_iter():
                    self._make_step(self.hard_negative_miner.cache, True)
                    self.hard_negative_miner.invalidate_cache()

            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})

            self.callbacks.on_batch_end(i)

            if self.step % self.save_every_steps == 0:
                self.metrics_collection.train_metrics = {k: v for k, v in meter.items()}
                self.estimator.model.eval()
                self.metrics_collection.val_metrics = self._run_validation(epoch, val_loader,
                                                                           foldername)

            self.callbacks.on_step_end(epoch, self.step, self.best_val_bce, self.best_val_dice)

        return {k: v / len(train_loader) for k, v in avg_meter.items()}

    def _save_validation_predict(self, foldername, name, fullpred, prefix=''):
        vispred_dir = os.path.join('./vis_predictions/', foldername)
        if not os.path.exists(vispred_dir):
            os.makedirs(vispred_dir, exist_ok=True)

        res_path_geo = os.path.join(vispred_dir, name)
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(res_path_geo, fullpred.shape[1], fullpred.shape[0], 1,
                                  gdal.GDT_Float32)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(fullpred)
        outRaster.FlushCache()

    def _run_validation(self, epoch, val_loader, foldername):
        avg_meter = defaultdict(float)

        pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                    desc="Epoch={}/step={}/valid".format(epoch, self.step), ncols=0)
        imagenames = []
        fullpreds = dict()
        fullbces = dict()
        fulldices = dict()
        for i, data in pbar:
            self.callbacks.on_batch_begin(i)
            image_names = data['image_name']
            meter, ypreds = self._make_step(data, False)
            for k, val in meter.items():
                avg_meter[k] += val

            # print('\nimage_names: {}'.format(image_names))
            for k in range(len(image_names)):
                fn = image_names[k].replace('.png', '')
                fullpred = ypreds[0, 0, :, :].cpu().numpy()

                imagenames.append(fn)
                fullpreds[fn] = fullpred
                fullbces[fn] = meter['bce']
                fulldices[fn] = meter['dice']

            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})

            self.callbacks.on_batch_end(i)

#        print('fullbces: {}'.format(fullbces))
#        print('self.best_val_bce: {}'.format(self.best_val_bce))
#        print('fulldices: {}'.format(fulldices))
#        print('self.best_val_dice: {}'.format(self.best_val_dice))
        for fn in imagenames:
            if fullbces[fn] < self.best_val_bce[fn]:
                self.best_val_bce[fn] = fullbces[fn]
                self._save_validation_predict(foldername, 'Best_BCE_'+fn + '.tif', fullpreds[fn])

            if fulldices[fn] > self.best_val_dice[fn]:
                self.best_val_dice[fn] = fulldices[fn]
                self._save_validation_predict(foldername, 'Best_DICE_'+fn + '.tif', fullpreds[fn])

        imagenames = None
        fullpreds = None
        fullbces = None
        fulldices = None
        gc.collect()

        return {k: v / len(val_loader) for k, v in avg_meter.items()}

    def _make_step(self, data, training):
        images = data['image']
        ytrues = data['mask']

        if training is True:
            meter, ypreds = self.estimator.make_step_itersize(images, ytrues, training,
                                                              self.metrics)
        else:
            meter, ypreds = self.estimator.make_step_itersize_large(images, ytrues, training,
                                                                    self.metrics)

        return meter, ypreds

    def fit(self, train_loader, val_loader, nb_epoch, foldername):
        self.callbacks.on_train_begin()

        self.step = self.estimator.start_step
        self.best_val_bce = self.estimator.best_val_bce
        self.best_val_dice = self.estimator.best_val_dice

        for epoch in range(self.estimator.start_epoch, nb_epoch):
            self.callbacks.on_epoch_begin(epoch, self.step)

            self.estimator.model.train()
            self.metrics_collection.train_metrics = self._run_training(epoch, train_loader,
                                                                       val_loader, foldername)
            self.estimator.model.eval()
            self.metrics_collection.val_metrics = self._run_validation(epoch, val_loader,
                                                                       foldername)

            self.callbacks.on_epoch_end(epoch, self.step, self.best_val_bce, self.best_val_dice)

            if self.metrics_collection.stop_training:
                break

        self.callbacks.on_train_end()


def train(ds, folds, config, num_workers=0, transforms=None, skip_folds=None,
          num_channels_changed=False):
    """
    here we construct all needed structures and specify parameters
    """
    os.makedirs(os.path.join(config.results_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, 'logs'), exist_ok=True)

    print('config.network: {}'.format(config.network))

    for fold, (train_idx, val_idx) in enumerate(folds):
        # train_idx = [train_idx[0]]
        if skip_folds and fold in skip_folds:
            continue

        save_path = os.path.join(config.results_dir, 'weights', config.folder)
        model = None
        if config.network == 'extensionunet':
            model = models[config.network](in_channels=config.num_channels, n_classes=1,
                                           nonlinearity='leaky_relu')
        elif config.network == 'resnet34':
            model = models[config.network](num_classes=1, num_channels=config.num_channels)
        elif config.network == 'denseunet':
            model = models[config.network](in_channels=config.num_channels, n_classes=1)

        estimator = Estimator(model, optimizers[config.optimizer], losses[config.loss], save_path,
                              iter_size=config.iter_size, test_iter_size=config.test_iter_size,
                              lr=config.lr, num_channels_changed=num_channels_changed)

        callbacks = [
            # StepLR(config.lr, num_epochs_per_decay=30, lr_decay_factor=0.1),
            InverseLR(config.lr, decay_rate=0.98, decay_steps=20),
            ModelSaver(1, ("fold"+str(fold)+"_best.pth"), best_only=True),
            CheckpointSaver(1, ("fold"+str(fold)+"_checkpoint.pth")),
            # EarlyStopper(10),
            TensorBoard(os.path.join(config.results_dir, 'logs', config.folder,
                        'fold{}'.format(fold)))
        ]

        # hard_neg_miner = HardNegativeMiner(rate=10)
        metrics = [('dice', dice), ('bce', binary_cross_entropy), ('dice round', dice_round)]
        # metrics = []
        trainer = PytorchTrain(estimator,
                               fold=fold,
                               metrics=metrics,
                               callbacks=callbacks,
                               hard_negative_miner=None)

        train_loader = PytorchDataLoader(TrainDataset(ds, train_idx, config, transforms=transforms),
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=num_workers,
                                         pin_memory=True)
        val_loader = PytorchDataLoader(ValDataset(ds, val_idx, config, transforms=None),
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       drop_last=False,
                                       num_workers=num_workers,
                                       pin_memory=True)

        trainer.fit(train_loader, val_loader, config.nb_epoch, config.folder)


def onetrain(trainds, valds, ntrain, nval, config, num_workers=0, transforms=None,
             num_channels_changed=False):
    """
    here we construct all needed structures and specify parameters
    """
    os.makedirs(os.path.join(config.results_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, 'logs'), exist_ok=True)

    train_idx = [i for i in range(ntrain)]
    val_idx = [i for i in range(nval)]

    print('config.network: {}'.format(config.network))

    save_path = os.path.join(config.results_dir, 'weights', config.folder)
    model = None
    if config.network == 'extensionunet':
        model = models[config.network](in_channels=config.num_channels, n_classes=1,
                                       nonlinearity='leaky_relu')
    elif config.network == 'resnet34':
        model = models[config.network](num_classes=1, num_channels=config.num_channels)
    elif config.network == 'denseunet':
        model = models[config.network](in_channels=config.num_channels, n_classes=1)

    estimator = Estimator(model, optimizers[config.optimizer], losses[config.loss], save_path,
                          iter_size=config.iter_size, test_iter_size=config.test_iter_size,
                          lr=config.lr, num_channels_changed=num_channels_changed)

    callbacks = [
        StepLR(config.lr, num_epochs_per_decay=30, lr_decay_factor=0.1),
        # InverseLR(config.lr, decay_rate=0.95, decay_steps=500),
        ModelSaver(1, 20, ("onetrain_best.pth"), best_only=True),
        CheckpointSaver(1, 20, ("onetrain_checkpoint.pth")),
        # EarlyStopper(10),
        TensorBoard(os.path.join(config.results_dir, 'logs', config.folder, 'onetrain'))
    ]

    # hard_neg_miner = HardNegativeMiner(rate=10)
    metrics = [('dice', dice), ('bce', binary_cross_entropy), ('dice round', dice_round)]
    # metrics = []
    trainer = PytorchTrain(estimator,
                           fold=-1,
                           metrics=metrics,
                           callbacks=callbacks,
                           hard_negative_miner=None)

    train_loader = PytorchDataLoader(TrainDataset(trainds, train_idx, config,
                                     transforms=transforms),
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
    val_loader = PytorchDataLoader(ValDataset(valds, val_idx, config, transforms=None),
                                   batch_size=config.test_batch_size,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=num_workers,
                                   pin_memory=True)

    print('ntrain : nval = {} : {}'.format(ntrain, nval))
    print('len(train_loader) : len(val_loader) = {} : {}'.format(len(train_loader),
          len(val_loader)))

    trainer.fit(train_loader, val_loader, config.nb_epoch, config.folder)
