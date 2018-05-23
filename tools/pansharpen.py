#!/usr/bin/env python

import argparse
import logging
import subprocess


def main(args):
    parser = argparse.ArgumentParser(
        description='Generate a pansharpen image using gdal')
    parser.add_argument("pan_dataset", help="Panchromatic dataset")
    parser.add_argument("spectral_dataset", help="Spectral dataset")
    parser.add_argument("out_dataset", help="Output dataset")
    parser.add_argument(
        "--to_rgb", action="store_true",
        help="Generate an out_dataset that is 8 bit per channel RGB image")
    parser.add_argument(
        "--blue",
        help="Blue band index. Typically bands are stored "
             "in increasing wave lengths, that is blue, green, red, infrared, etc."
             " If its not specified, it defaults to 1")
    parser.add_argument(
        "--green", help="Green band index."
                        " If its not specified, it defaults to blue index + 1")
    parser.add_argument(
        "--red", help="Red band index."
                      " If its not specified, it defaults to green index + 1")
    parser.add_argument(
        "--to_8bit", action="store_true",
        help="Generate an out_dataset with 8 bit per band")
    args = parser.parse_args(args)

    if (not args.to_rgb and (args.red or args.green or args.blue)):
        raise RuntimeError("Error: you can specify color band indexes "
                           "only when to_rgb is specified")

    call_args_rest = []
    out_dataset = args.out_dataset
    if (args.to_rgb or args.to_8bit):
        out_dataset = "_%s" % args.out_dataset
        if (args.to_rgb):
            # convert to RGB
            blue_index = 1
            if args.blue:
                blue_index = int(args.blue)
            green_index = blue_index + 1
            if args.green:
                green_index = int(args.green)
            red_index = green_index + 1
            if args.red:
                red_index = int(args.red)
            bands = [red_index, green_index, blue_index]
            # pansharpen and reorder RGB bands
            call_args_rest = [v for band in bands for v in ["-b", str(band)]]

    call_args = ["gdal_pansharpen.py",
                 args.pan_dataset, args.spectral_dataset,
                 out_dataset] + call_args_rest
    print("\nPansharpening ...")
    print(call_args)
    subprocess.call(call_args)

    if (args.to_rgb or args.to_8bit):
        call_args_8bit = []
        message = "\nConverting to "
        if (args.to_8bit):
            call_args_8bit = ["-ot", "Byte", "-scale"]
            message += "8 bit "
        call_args_rgb = []
        if (args.to_rgb):
            call_args_rgb = ["-b", "1", "-b", "2", "-b", "3",
                             "-co", "PHOTOMETRIC=RGB"]
            message += "RGB "
        message += "..."
        call_args = ["gdal_translate", out_dataset, args.out_dataset,
                     "-co", "COMPRESS=DEFLATE"] + call_args_8bit + call_args_rgb
        print(message)
        print(call_args)
        subprocess.call(call_args)


if __name__ == '__main__':
    import sys
    try:
        main(sys.argv[1:])
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
