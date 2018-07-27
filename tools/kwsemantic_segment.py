import os
import sys
import numpy as np
from osgeo import gdal
import argparse
import json

from danesfield.segmentation.semantic.utils.utils import update_config
from danesfield.segmentation.semantic.tasks.seval import Evaluator
from danesfield.segmentation.semantic.utils.config import Config

# Need to append to sys.path here as the pretrained model includes an
# import statement for "models" rather than
# "danesfield.segmentation.semantic.models"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../danesfield/segmentation/semantic"))

parser = argparse.ArgumentParser(description='configuration for semantic segmantation task.')
parser.add_argument('config_path', help='configuration file path.')
parser.add_argument('pretrain_model_path', help='pretrained model file path.')
parser.add_argument('rgbpath', help='3-band 8-bit RGB image path')
parser.add_argument('dsmpath', help='1-band float DSM file path')
parser.add_argument('dtmpath', help='1-band float DTM file path')
parser.add_argument('msipath', help='8-band float MSI file path')
parser.add_argument('outfname', help='out filename for prediction probability and class mask')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    cfg = json.load(f)
    pretrain_model_path = args.pretrain_model_path
    rgbpath = args.rgbpath
    dsmpath = args.dsmpath
    dtmpath = args.dtmpath
    msipath = args.msipath
    outfname = args.outfname
    cfg['pretrain_model_path'] = pretrain_model_path
    cfg['out_fname'] = outfname
config = Config(**cfg)


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def predict():
    img_data = np.transpose(gdal.Open(rgbpath).ReadAsArray(), (1, 2, 0))

    dsm_data = gdal.Open(dsmpath).ReadAsArray()
    dtm_data = gdal.Open(dtmpath).ReadAsArray()
    ndsm_data = dsm_data - dtm_data
    ndsm_data[ndsm_data < 0] = 0
    ndsm_data[ndsm_data > 40] = 40
    ndsm_data = ndsm_data/40*255

    msi_image = np.transpose(gdal.Open(msipath).ReadAsArray(), (1, 2, 0))
    red_map = msi_image[:, :, 4].astype(np.float)
    nir_map = msi_image[:, :, 6].astype(np.float)

    ndvi = (nir_map - red_map)/(nir_map + red_map + 1e-7)
    ndvi[ndvi < 0] = 0
    ndvi[ndvi > 1] = 1
    ndvi_data = ndvi*255.0

    input_data = np.moveaxis(np.dstack([img_data, ndsm_data, ndvi_data])/255, -1, 0)
    input_data = input_data.astype(np.float32)
    input_data = (input_data - 0.5)*2

    keval = Evaluator(config)
    keval.onepredict(input_data, dsmpath, outfname)


if __name__ == "__main__":
    config = update_config(config, img_rows=2048, img_cols=2048, target_rows=2048,
                           target_cols=2048, num_channels=5)
    predict()
