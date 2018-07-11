import os
import numpy as np
import cv2
from .abstract_image_type import AbstractImageType


class MultibandImageType(AbstractImageType):
    """
    image type, that has dem/dtm information
    """
    def __init__(self, paths, fn, border, has_alpha):
        super().__init__(paths, fn, has_alpha)
        self.border = border
        self.img_data = None
        self.ndsm_data = None
        self.ndvi_data = None
        self.gtl_data = None

    def read_image(self):
        if self.img_data is None:
            self.img_data = cv2.imread(os.path.join(self.paths['images'], self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.ndsm_data is None:
            self.ndsm_data = cv2.imread(os.path.join(self.paths['ndsms'], self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.ndvi_data is None:
            self.ndvi_data = cv2.imread(os.path.join(self.paths['ndvis'], self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        img_data = self.img_data
        ndsm_data = self.ndsm_data
        ndvi_data = self.ndvi_data
#        img_data = (np.float32(self.img_data.copy())/255.0 - 0.5)*2.0
#        ndsm_data = (np.float32(self.ndsm_data.copy())/255.0 - 0.5)*2.0
#        ndvi_data = (np.float32(self.ndvi_data.copy())/255.0 - 0.5)*2.0

        return self.finalyze(np.dstack([img_data, ndsm_data, ndvi_data]))

    def read_mask(self):
        if self.gtl_data is None:
            self.gtl_data = cv2.imread(os.path.join(self.paths['masks'], self.fn), cv2.IMREAD_UNCHANGED)
        mask = (self.gtl_data == 6).astype(np.uint8) * 255
        return self.finalyze(mask)

    def read_alpha(self):
        if self.img_data is None:
            self.img_data = cv2.imread(os.path.join(self.paths['images'], self.fn), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        return self.finalyze(self.img_data[:,:,3])

    def finalyze(self, data):
        return self.reflect_border(data, b=self.border)

