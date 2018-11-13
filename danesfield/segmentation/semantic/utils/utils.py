###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

import numpy as np
import cv2
from sklearn.model_selection import KFold
from .config import Config


def heatmap(map):
    map = (map*255).astype(np.uint8)
    return cv2.applyColorMap(map, cv2.COLORMAP_BONE)


def get_folds(data, num):
    kf = KFold(n_splits=num, shuffle=True, random_state=42)
    kf.get_n_splits(data)
    return kf.split(data)


def update_config(config, **kwargs):
    d = config._asdict()
    d.update(**kwargs)
    print(d)
    return Config(**d)
