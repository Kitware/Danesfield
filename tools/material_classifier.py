#!/usr/bin/env python

import argparse
import logging
import sys
import os

from danesfield.materials.pixel_prediction.util.model import Classifier
from danesfield.materials.pixel_prediction.util.misc import save_output


def main(args):
    parser = argparse.ArgumentParser(
        description='Classify materials for an orthorectifed image.')
    parser.add_argument('image_path', help='Path to image file.')
    parser.add_argument(
        'imd_path', help='Path to image metadata (.IMD) file.')
    parser.add_argument(
        'output_path', help='Path to save result images to.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU. (May have to adjust batch_size value)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of pixels classified at a time.')
    parser.add_argument('--aux_out', action='store_true',
                        help='Output a class probably image in output path for each image.')
    args = parser.parse_args(args)

    # Load model and select option to use CUDA
    classifier = Classifier(args.cuda, args.batch_size)

    # Use CNN to create material map
    material_output, debug_ouput = classifier.Evaluate(
        args.image_path, args.imd_path)

    # Save results
    output_path = args.output_path + \
        os.path.splitext(os.path.split(args.image_path)[1])[0]

    save_output(material_output, output_path)
    if args.aux_out:
        save_output(debug_ouput, output_path, aux=True)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
