import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from ..util import misc
from ..architecture import ResNet as RN
from PIL import Image
import os
import pickle
from osgeo import gdal
from collections import OrderedDict


class Classifier():
    def __init__(self, cuda, output_path, batch_size=256,
                 model_path='best_model.pth.tar', aux_out=False):
        # Model
        self.model = RN.resnet18(num_classes=12)

        # Load the weights from the saved network
        model_path = "/../architecture/RN18_All.pth.tar"
        checkpoint = torch.load(os.path.dirname(__file__) + model_path,
                                map_location=lambda storage,
                                loc: storage)['state_dict']

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

        # Folder where to save output
        self.test_dir = output_path

        # If true will output debug image
        self.aux_out = aux_out

    def Evaluate(self, img_path, imd_path):
        # Load image and convert to tensor
        img = np.transpose(
            gdal.Open(img_path, gdal.GA_ReadOnly).ReadAsArray(), (1, 2, 0))

        mask = misc.get_mask(img)
        img = misc.calibrate_img(img, imd_path)
        tensor_img = misc.image_to_tensor(img)
        num_test = img.shape[0]*img.shape[1]

        # Make a dataloader object out of image in order to put in CNN
        data_loader = misc.getTestDataLoader(
            tensor_img, self.batch_size)

        # Set model to evaulate (need this for batch_norm layers)
        self.model.eval()
        result = np.array([], dtype=np.int64).reshape(0, 1)
        confidence = np.array([], dtype=np.int64).reshape(0, 1)
        out_p = np.array([], dtype=np.int64).reshape(0, 12)
        for i, data in enumerate(data_loader, 0):
            print("Evaluate: {0:3.1f}% ".format(
                i / num_test * 100 * self.batch_size), end="\r")
            sig = data

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

            confid = confidence_val.cpu().numpy()
            confid = np.reshape(confid, [confid.shape[0], 1])
            confidence = np.vstack((confidence, confid))

            class_probs = out_prob.data.cpu().numpy()
            class_probs = np.reshape(
                class_probs, [class_probs.shape[0], 12])
            out_p = np.vstack((out_p, class_probs))

        print(" "*100, end="\r")

        filename = os.path.split(img_path)[1][:-4]
        result_img = np.reshape(result, [img.shape[0], img.shape[1]])
        confidence_img = np.reshape(
            confidence, [img.shape[0], img.shape[1]])
        output_prob_img = np.reshape(
            out_p, [img.shape[0], img.shape[1], 12])
        result_img = misc.apply_mask(result_img, mask)

        Image.fromarray(result_img.astype('uint8')).save(
            self.test_dir+'/'+filename+'_material_map.tif')
        Image.fromarray(misc.ColorImage(result_img).astype('uint8')).save(
            self.test_dir+'/'+filename+'_result_color.png')
        if self.aux_out is True:
            Image.fromarray(confidence_img.astype('uint8')).save(
                self.test_dir+'/'+filename+'_confidence.tif')
            pickle.dump(output_prob_img, open(
                self.test_dir+'/'+filename+'_outprobs.p', "wb"))
        return result_img
