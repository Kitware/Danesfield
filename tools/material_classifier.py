#!/usr/bin/env python
import argparse
import logging
import os
import sys

from danesfield.materials.pixel_prediction.util.model import Classifier
from danesfield.materials.pixel_prediction.util.misc import save_output, Combine_Result, transfer_metadata  # noqa: E501


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

    # Load model and select option to use CUDA
    classifier = Classifier(args.cuda, args.batch_size, args.model_path)

    # Use CNN to create material map
    if len(args.image_paths) != len(args.info_paths):
        raise IOError(
            ('The number of image paths {}, '
             'does not match the number of metadata paths {}.').format(len(args.image_paths),
                                                                       len(args.info_paths)))

    combine_result = Combine_Result('max_prob')
    for image_path, info_path in zip(args.image_paths, args.info_paths):
        _, prob_output = classifier.Evaluate(image_path, info_path)
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

    transfer_metadata(output_path, args.image_paths[0])


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
