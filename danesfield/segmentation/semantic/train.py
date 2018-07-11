import torch
import os

from tasks.transforms import augment_flips_color

from dataset.image_provider import ImageProvider
from dataset.threeband_image import ThreebandImageType
from dataset.multiband_image import MultibandImageType
from utils.utils import get_folds, update_config
from tasks.train import train
from tasks.train import onetrain
import argparse
import json
from utils.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('train_data_path')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    cfg = json.load(f)
    train_data_path = args.train_data_path
    print('train_data_path: {}'.format(train_data_path))
    dataset_path, train_dir = os.path.split(train_data_path)
    print('dataset_path: {}'.format(dataset_path) +
          ',  train_dir: {}'.format(train_dir))
    cfg['dataset_path'] = dataset_path
config = Config(**cfg)

paths = {
    'masks': train_data_path + '/gtl',
    'images': train_data_path + '/rgb',
    'ndsms': train_data_path + '/ndsm',
    'ndvis': train_data_path + '/ndvi',
}

paths = {k: os.path.join(config.dataset_path, v) for k, v in paths.items()}
valpaths = {k: os.path.join(config.dataset_path, v).replace(
    'DaytonJacksonville', '4AOIs') for k, v in paths.items()}


def single_train(config):
    print('paths: {}'.format(paths))
    print('valpaths: {}'.format(valpaths))

    num_epochs = config.nb_epoch
    num_workers = 0 if os.name == 'nt' else 8

    ntrain = len(os.listdir(paths['images']))
    nval = len(os.listdir(valpaths['images']))

    # 1st stage: train on RGB three channels.
    trainds = ImageProvider(ThreebandImageType, paths, image_suffix='.png')
    valds = ImageProvider(ThreebandImageType, valpaths, image_suffix='.png')
    config = update_config(config, num_channels=3, nb_epoch=5)
    # onetrain(trainds, valds, ntrain, nval, config,
    #          num_workers=num_workers, transforms=augment_flips_color)

    # 2nd stage: train on RGB+ndsm+ndvi.
    trainds = ImageProvider(MultibandImageType, paths, image_suffix='.png')
    valds = ImageProvider(MultibandImageType, valpaths, image_suffix='.png')
    config = update_config(config, num_channels=5, nb_epoch=num_epochs)
    onetrain(trainds, valds, ntrain, nval, config,
             num_workers=num_workers, transforms=augment_flips_color)
    # onetrain(trainds, valds, ntrain, nval, config,
    #          num_workers=num_workers, transforms=augment_flips_color,
    #          num_channels_changed=True)

    # 3rd stage: change the loss function.
    config = update_config(config, loss=config.loss +
                           '_w', nb_epoch=num_epochs + num_epochs)
    onetrain(trainds, valds, ntrain, nval, config,
             num_workers=num_workers, transforms=augment_flips_color)


def multifold_cross_train(config):
    print('paths: {}'.format(paths))
    print('valpaths: {}'.format(valpaths))

    num_epochs = config.nb_epoch
    num_workers = 0 if os.name == 'nt' else 8

    # 1st stage: train on RGB three channels.
    ds = ImageProvider(ThreebandImageType, paths, image_suffix='.png')
    folds = get_folds(ds, 5)
    config = update_config(config, num_channels=3, nb_epoch=5)
    train(ds, folds, config, num_workers=num_workers,
          transforms=augment_flips_color)

    # 2nd stage: train on RGB+ndsm+ndvi.
    config = update_config(config, num_channels=5, nb_epoch=num_epochs)
    ds = ImageProvider(MultibandImageType, paths, image_suffix='.png')
    train(ds, folds, config, num_workers=num_workers,
          transforms=augment_flips_color, num_channels_changed=True)

    # 3rd stage: change the loss function.
    config = update_config(config, loss=config.loss +
                           '_w', nb_epoch=num_epochs + num_epochs)
    train(ds, folds, config, num_workers=num_workers,
          transforms=augment_flips_color)


if __name__ == "__main__":
    single_train(config)
    # multifold_cross_train(config)
