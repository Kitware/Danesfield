#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import argparse
import logging
import os
import sys

from danesfield.materials.pixel_prediction.util.model import Classifier
from danesfield.materials.pixel_prediction.util.misc import save_output, Combine_Result, transfer_metadata, order_images  # noqa: E501


def main(args):
    parser = argparse.ArgumentParser(description='Classify materials in an orthorectifed image.')

    parser.add_argument('--image_paths', nargs='*', required=True,
                        help='List of image paths. Images should be orthorectified.')

    parser.add_argument('--info_paths', nargs='*', required=True,
                        help='List of metadata files for image files. (.tar or .imd)')

    parser.add_argument('--output_dir', required=True,
                        help='Directory where result images will be saved.')

    parser.add_argument('--model_path',
                        help='Path to model used for evaluation.')

    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU. (May have to adjust batch_size value)')

    parser.add_argument('--batch_size', type=int, default=40000,
                        help='Number of pixels classified at a time.')

    parser.add_argument('--outfile_prefix', type=str,
                        help='Output filename prefix.')

    args = parser.parse_args(args)

    # Use CNN to create material map
    if len(args.image_paths) != len(args.info_paths):
        raise IOError(
            ('The number of image paths {}, '
             'does not match the number of metadata paths {}.').format(len(args.image_paths),
                                                                       len(args.info_paths)))

    # Object that merges results
    combine_result = Combine_Result('max_prob')

    # Order image paths and metadata
    image_paths, info_paths = order_images(args.image_paths, args.info_paths)
    num_images = len(image_paths)

    # Load model and select option to use CUDA
    classifier = Classifier(image_paths, args.model_path, batch_size=args.batch_size)

    model_name = os.path.split(args.model_path)[1]
    img_per_set = int(model_name[9:11])

    # Use different model based on the number of images given
    if img_per_set == 1:
        for i, (image_path, info_path) in enumerate(zip(image_paths, info_paths)):
            print('Material classification: {0:2d}/{1:2d}'.format(i+1, num_images))
            prob_output = classifier.Evaluate([image_path], [info_path])
            combine_result.update(prob_output)
    else:
        N = 10  # Number of random samples taken
        for i in range(N):
            print('Material classification: {0:2d}/{1:2d}'.format(i+1, N))
            prob_output = classifier.Evaluate(image_paths, info_paths)
            combine_result.update(prob_output)

    # Save results
    if args.outfile_prefix:
        output_file_basename = '{}_MTL'.format(args.outfile_prefix)
    else:
        # This is the old / default output file basename
        output_file_basename = 'max_prob'

    output_path = os.path.join(args.output_dir, output_file_basename + '.tif')

    combined_result = combine_result.call()

    save_output(combined_result, output_path)

    transfer_metadata(output_path, image_paths[0])


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
