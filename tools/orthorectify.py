#!/usr/bin/env python

from danesfield import ortho

import argparse
import logging


def main(args):
    parser = argparse.ArgumentParser(
        description='Orthorectify an image given the DSM')
    parser.add_argument("source_image", help="Source image file name")
    parser.add_argument("dsm", help="Digital surface model (DSM) image file name")
    parser.add_argument("destination_image", help="Orthorectified image file name")
    parser.add_argument('-t', "--occlusion-thresh", type=float, default=1.0,
                        help="Threshold on height difference for detecting "
                        "and masking occluded regions (in meters)")
    parser.add_argument('-d', "--denoise-radius", type=float, default=2,
                        help="Apply morphological operations with this radius "
                        "to the DSM reduce speckled noise")
    parser.add_argument("--raytheon-rpc", type=str,
                        help="Raytheon RPC file name. If not provided, "
                        "the RPC is read from the source_image")
    parser.add_argument("--dtm", type=str,
                        help="Optional DTM parameter used to replace nodata areas in the "
                             "orthorectified image")
    args = parser.parse_args(args)

    ret = ortho.orthorectify(args.source_image, args.dsm, args.destination_image,
                             args.occlusion_thresh, args.denoise_radius,
                             args.raytheon_rpc, args.dtm)
    if ret == ortho.ERROR:
        raise RuntimeError("Error: orthorectification failed")


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
