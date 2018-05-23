#!/usr/bin/env python

import argparse
import logging
import sys

from danesfield.materials.pixel_prediction.util.model import Classifier


def main(args):
    parser = argparse.ArgumentParser(description='Classify materials for an orthorectifed image.')
    parser.add_argument('image_path', help='Path to image file.')
    parser.add_argument('imd_path', help='Path to image metadata (.IMD) file.')
    parser.add_argument('output_path', help='Path to save result images to.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU. (May have to adjust batch_size value)')
    parser.add_argument('--aux_out', action='store_true',
                        help='Output a class probably image in output path for each image.')
    args = parser.parse_args(args)

    # Load model and select option to use CUDA
    classifier = Classifier(args.cuda, args.output_path, batch_size=2**15, aux_out=args.aux_out)

    # Use CNN to create material map
    classifier.Evaluate(args.image_path, args.imd_path)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
