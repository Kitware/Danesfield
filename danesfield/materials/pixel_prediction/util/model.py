import torch
import numpy as np
import gdal
from scipy.ndimage import zoom
import os

from ..util import misc
from ..architecture import ResNet as RN


def subsample_image(image, factor):
    return image[::factor, ::factor]


def upsample_image(image, factor):
    upscale_image = np.zeros((image.shape[0]*factor, image.shape[1]*factor, image.shape[2]))
    for channel in range(image.shape[2]):
        upscale_image[:, :, channel] = zoom(image[:, :, channel], factor, order=0)
    return upscale_image


class Classifier():
    def __init__(self, image_paths, model_path, batch_size=20000, subfactor=2):
        # Hyperparameter(s)
        self.batch_size = batch_size
        self.subfactor = subfactor
        self.num_classes = 12

        num_images = len(image_paths)

        model_name = os.path.split(model_path)[1]
        self.images_per_set = int(model_name[9:11])

        if num_images < self.images_per_set:
            raise IOError('Cannot use model {} with less than {} images.'
                          'Only {} images are given.'.format(model_name,
                                                             self.images_per_set,
                                                             num_images))

        if self.images_per_set == 1:
            # Use single-image algorithm
            self.model = RN.model_A(num_classes=self.num_classes)
            self.model = torch.nn.DataParallel(self.model)
            self.model_type = 'A'
        else:
            # Use multi-image random sampling algorithm
            self.model = RN.model_B(num_classes=self.num_classes)
            self.model = torch.nn.DataParallel(self.model)
            self.model_type = 'B'

        # Load the weights from the saved network
        self.device = torch.device('cuda')
        checkpoint = torch.load(model_path)['state_dict']
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)

        # Load mean and std values from model
        self.dataset_stats = misc.get_train_data_stats(model_path)

        # Get coordinate set for tiling images
        self.coordinates = misc.coordinate_set_generator(image_paths[0], self.subfactor)
        dst = gdal.Open(image_paths[0])
        self.height, self.width = dst.RasterXSize, dst.RasterYSize

    def Evaluate(self, image_set, info_set):
        if self.model_type == 'A':
            sub_image_paths = image_set
        elif self.model_type == 'B':
            # Take a random sample of image paths
            image_indices = np.random.randint(len(image_set), size=self.images_per_set)
            sub_image_paths = np.take(image_set, image_indices)
            info_set = np.take(info_set, image_indices)

        final_result = np.zeros((self.width // self.subfactor, self.height //
                                 self.subfactor, self.num_classes), dtype=float)

        # For each set of coordinates
        for c, coord in enumerate(self.coordinates):
            x0, y0, x1, y1 = coord

            stack_sub_img = np.zeros((y1, x1, 8*len(sub_image_paths)))
            for i, (img_path, info_path) in enumerate(zip(sub_image_paths, info_set)):
                x0s, y0s = x0 * self.subfactor, y0 * self.subfactor
                x1s, y1s = x1 * self.subfactor, y1 * self.subfactor
                sub_img = np.transpose(gdal.Open(img_path).ReadAsArray(
                    x0s, y0s, x1s, y1s), (1, 2, 0))
                sub_img = subsample_image(sub_img, self.subfactor)
                sub_img = misc.calibrate_img(sub_img.astype(float), info_path)
                stack_sub_img[:, :, i*8:(i+1)*8] = sub_img

            # If test then classify the sub_img_stack
            dataloader = misc.create_dataloader(stack_sub_img, self.dataset_stats, self.batch_size)
            result = self._neural_network(dataloader, self.model, self.device)
            result = np.reshape(result, [y1, x1, 12])

            # Put results in final material map
            final_result[y0:y0+y1, x0:x0+x1, :] += result

        final_result = upsample_image(final_result, self.subfactor)
        return final_result

    def _neural_network(self, dataloader, model, device):
        self.model.eval()
        conf_img = torch.Tensor().float()
        conf_img = conf_img.to(device)

        with torch.no_grad():
            SM = torch.nn.Softmax(dim=1)
            for i, batch in enumerate(dataloader):
                data = batch

                data_v = data.to(device)
                output = model(data_v)
                out_prob = SM(output)

                del data_v, data

                # Generate confidence map
                conf_img = torch.cat((conf_img, out_prob))

        conf_vector = conf_img.cpu().numpy()
        return conf_vector
