import os
from dataset.image_provider import ImageProvider
# from dataset.threeband_image import ThreebandImageType
from dataset.multiband_image import MultibandImageType
# from utils.utils import get_folds
from utils.utils import update_config
from tasks.concrete_eval import GdalFullEvaluator
import argparse
import json
from utils.config import Config
from utils.merge_preds import merge_onetrain_tiffs
from utils.make_submission import make_submission

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('pretrain_model_path')
parser.add_argument('test_data_path')
parser.add_argument('dsm_tifdata_path')
parser.add_argument('output_file')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    cfg = json.load(f)
    pretrain_model_path = args.pretrain_model_path
    dsm_tifdata_path = args.dsm_tifdata_path
    test_data_path = args.test_data_path
    out_file = args.output_file
    dataset_path, test_dir = os.path.split(test_data_path)
    cfg['dataset_path'] = dataset_path
    cfg['pretrain_model_path'] = pretrain_model_path
    cfg['dsm_tifdata_path'] = dsm_tifdata_path
config = Config(**cfg)

paths = {
    'masks': test_data_path + '/gtl',
    'images': test_data_path + '/rgb',
    'ndsms': test_data_path + '/ndsm',
    'ndvis': test_data_path + '/ndvi',
}

# paths = {k: os.path.join(config.dataset_path, v) for k,v in paths.items()}
paths = {k: v for k, v in paths.items()}

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def predict():
    """
    heat up weights for 5 epochs
    """
    print('paths: {}'.format(paths))

    ds = ImageProvider(MultibandImageType, paths, border=0, image_suffix='.png')

    print('ds: {}'.format(ds))

    folds = [([], list(range(len(ds)))) for i in range(5)]
    num_workers = 0 if os.name == 'nt' else 8
    keval = GdalFullEvaluator(config, ds, folds, test=True, num_workers=num_workers, border=0)
    keval.onepredict()


if __name__ == "__main__":
    # config = update_config(config, img_rows=1024, img_cols=1024, target_rows=1024,
    #                        target_cols=1024, num_channels=5)
    config = update_config(config, img_rows=2048, img_cols=2048, target_rows=2048,
                           target_cols=2048, num_channels=5)
    print("predict stage 1/3")
    predict()

    print("predict stage 2/3")
    test_dsmdata_dir = config.dsm_tifdata_path
    merge_onetrain_tiffs(os.path.join(config.results_dir, 'results', config.folder),
                         test_dsmdata_dir)

    print("predict stage 3/3")
    make_submission(os.path.join(config.results_dir, 'results', config.folder, 'merged'),
                    test_dsmdata_dir, out_file)
