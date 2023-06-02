import sys
import cv2 as cv
from glob import glob
import os
import numpy as np

KERNEL_SIZE = 5

def main(args):
    texture_path = args[0]
    backup_path = os.path.join(texture_path, "undialated_textures")

    # Make the backup path
    os.mkdir(backup_path)

    for fpath in glob(texture_path + "/*.png"):
        tmg = cv.imread(fpath)
        
        # Backup the image at backup_path
        fname = os.path.basename(fpath)
        cv.imwrite(backup_path + "/" + fname, tmg)

        # Dialate the texture
        krn = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=tmg.dtype)
        dmg = cv.dilate(tmg, krn)

        # Put the original texture back into the dialated one
        _, mask = cv.threshold(tmg, 0, 255, cv.THRESH_BINARY)
        mask = cv.bitwise_not(mask)
        dmg = cv.bitwise_and(dmg, mask)
        dmg = cv.add(tmg, dmg)

        # Overwrite the image with the dialated one
        cv.imwrite(texture_path + '/' + fname, dmg)

        # cv.namedWindow("test")
        # cv.imshow("test", dmg)
        # while cv.waitKey(0) != 27:
            # pass
        # cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])