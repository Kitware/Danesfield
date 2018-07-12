import random
# import cv2
import numpy as np
from matplotlib import pyplot as plt

from tasks.transforms import ToTensor
from .image_provider import AbstractImageProvider
from .image_cropper import ImageCropper


class Dataset:
    """
    base class for pytorch datasets
    """

    def __init__(self, image_provider: AbstractImageProvider,
                 image_indexes, config, stage='train',
                 transforms=ToTensor()):
        self.cropper = ImageCropper(config.target_rows,
                                    config.target_cols,
                                    config.train_pad if stage == 'train' else config.test_pad,
                                    use_crop=True if stage == 'train' else False)
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(
            image_indexes, list) else image_indexes.tolist()
        if stage != 'train' and len(self.image_indexes) % 2:  # todo bugreport it
            self.image_indexes += [self.image_indexes[-1]]
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.config = config
        self.transforms = transforms
        if transforms is None:
            self.transforms = ToTensor()

    def __getitem__(self, item):
        raise NotImplementedError


class TrainDataset(Dataset):
    """
    dataset for train stage
    """

    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=ToTensor()):
        super(TrainDataset, self).__init__(image_provider,
                                           image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx):
        im_idx = self.image_indexes[idx % len(self.image_indexes)]

        item = self.image_provider[im_idx]
        sx, sy = self.cropper.random_crop_coords(item.image)
        if self.cropper.use_crop and self.image_provider.has_alpha:
            for i in range(10):
                alpha = self.cropper.crop_image(item.alpha, sx, sy)
                if np.mean(alpha) > 5:
                    break
                sx, sy = self.cropper.random_crop_coords(item.image)
            else:
                return self.__getitem__(random.randint(0, len(self.image_indexes)))

        im = self.cropper.crop_image(item.image, sx, sy)
        mask = self.cropper.crop_image(item.mask, sx, sy)

        # print('im.shape: {}'.format(im.shape)
        #        + ' <- item.image.shape: {}'.format(item.image.shape)
        #        + ', (x, y) = {}'.format((sx, sy)))

        im, mask = self.transforms(im, mask)

        cropim = np.transpose(im, (1, 2, 0))

        bcheck_input = False
        if bcheck_input:
            rgb = cropim[:, :, 0:3]
            ndsm = cropim[:, :, 3]
            ndvi = cropim[:, :, 4]
            gmask = np.squeeze(mask)

            rgb[gmask == 1, 0] = rgb[gmask == 1, 0]*0.8
            rgb[gmask == 1, 1] = rgb[gmask == 1, 1]*0.8 + 255*0.2
            rgb[gmask == 1, 2] = rgb[gmask == 1, 2]*0.8

            fig = plt.figure(figsize=(19, 11))
            a = fig.add_subplot(1, 3, 1)
            plt.imshow(rgb)
            plt.colorbar()
            a.set_title('rgb')

            b = fig.add_subplot(1, 3, 2)
            plt.imshow(ndsm)
            plt.colorbar()
            b.set_title('ndsm')

            c = fig.add_subplot(1, 3, 3)
            plt.imshow(ndvi)
            plt.colorbar()
            c.set_title('ndvi')

            figname = 'checktraindata/train_' + \
                str(im_idx) + '_sxsy_' + str(sx) + '_' + str(sy) + '.png'
            plt.savefig(figname)
            plt.close()

        return {'image': im, 'mask': mask, 'image_name': item.fn}

    def __len__(self):
        # epoch size is len images
        return len(self.image_indexes) * max(self.config.epoch_size, 1)


class SequentialDataset(Dataset):
    """
    dataset for inference
    """

    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=ToTensor()):
        super(SequentialDataset, self).__init__(image_provider,
                                                image_indexes, config, stage, transforms=transforms)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys.update({'sy', 'sx'})

    def init_good_tiles(self):
        self.good_tiles = []
        for im_idx in self.image_indexes:
            item = self.image_provider[im_idx]

            positions = self.cropper.cropper_positions(item.image)

            if self.image_provider.has_alpha:
                item = self.image_provider[im_idx]
                alpha_generator = self.cropper.sequential_crops(item.alpha)
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        item = self.image_provider[im_idx]

        im = self.cropper.crop_image(item.image, sx, sy)

        im = self.transforms(im)
        return {'image': im, 'startx': sx, 'starty': sy, 'image_name': item.fn}

    def __len__(self):
        return len(self.good_tiles)


class ValDataset(SequentialDataset):
    """
    dataset for validation
    """

    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=ToTensor()):
        super(ValDataset, self).__init__(image_provider,
                                         image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx):
        im_idx, sy, sx = self.good_tiles[idx]
        item = self.image_provider[im_idx]

        im = self.cropper.crop_image(item.image, sx, sy)
        mask = self.cropper.crop_image(item.mask, sx, sy)
        # print('im.shape: {}'.format(im.shape)
        #        + ' <- item.image.shape: {}'.format(item.image.shape)
        #        + ', (x, y) = {}'.format((sx, sy)))
        # cv2.imshow('w', im[...,:3])
        # cv2.imshow('m', mask)
        # cv2.waitKey()
        im, mask = self.transforms(im, mask)
        return {'image': im, 'mask': mask, 'startx': sx, 'starty': sy, 'image_name': item.fn}
