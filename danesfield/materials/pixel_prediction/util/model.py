import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from ..util import misc
from ..architecture import ResNet as RN
import os
# from osgeo import gdal
from collections import OrderedDict


class Classifier():
    def __init__(self, cuda, batch_size,
                 model_path="/../architecture/RN18_All.pth.tar"):
        # Model
        self.model = RN.resnet18(num_classes=12)

        # Load the weights from the saved network
        checkpoint = torch.load(os.path.dirname(__file__) + model_path,
                                map_location=lambda storage,
                                loc: storage)['state_dict']

        # Alter model weights to run on CPU
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)

        self.cuda = cuda

        if self.cuda is True:
            # If this is really slow, update your cuda version
            self.model = self.model.cuda()

        # Hyperparameter
        self.batch_size = batch_size

    def Evaluate(self, img_path, imd_path):
        # Caibrate image
        img = misc.calibrate_img(img_path, imd_path)

        # Make a dataloader object out of image in order to put in CNN
        loader = misc.getTestDataLoader(
            img, self.batch_size)

        # Set model to evaulate (required for batch_norm layers)
        self.model.eval()
        result = np.array([], dtype=np.int64).reshape(0, 1)
        out_probs = np.array([], dtype=np.int64).reshape(0, 12)

        for i, sig in enumerate(loader, 0):
            print("Evaluate: {0:4d}/{1:4d}".format(
                i+1, len(loader)), end="\r")

            if self.cuda is True:
                sig = sig.cuda()

            sigv = Variable(sig, volatile=True)

            UP = nn.Upsample(scale_factor=4, mode='linear')
            sigv = UP(sigv)

            out_prob = self.model(sigv)
            confidence_val, predicted = torch.max(out_prob.data, 1)

            labels = predicted.cpu().numpy()
            labels = np.reshape(labels, [labels.shape[0], 1])
            result = np.vstack((result, labels))

            class_probs = out_prob.data.cpu().numpy()
            class_probs = np.reshape(
                class_probs, [class_probs.shape[0], 12])
            out_probs = np.vstack((out_probs, class_probs))

        print(" "*100, end="\r")

        result_img = np.reshape(result, [img.shape[0], img.shape[1]])
        output_prob_img = np.reshape(
            out_probs, [img.shape[0], img.shape[1], 12])

        return result_img, output_prob_img
