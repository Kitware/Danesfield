import pickle
import os
import numpy as np


class RoofDataset():
    def __init__(self, root, num_category, npoints=3500, split='train', appendix=""):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.num_category = num_category
        self.data_filename = os.path.join(
            self.root, 'roof_seg_%s%s.pickle' % (split, appendix))
        print("data_filename", self.data_filename)
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
            self.filename_list = pickle.load(fp)

    def __getitem__(self, index):
        space_l = 3.0
        space_w = 3.0
        space_h = 3.0
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index]
        # print "semantic_seg.shape", semantic_seg.shape
        coordmax = np.max(point_set, axis=0)
        # print "coordmax", coordmax
        coordmin = np.min(point_set, axis=0)
        # print "coordmin", coordmin
        isvalid = False
        for i in range(10):
            # curcenter = point_set[np.random.choice(point_set.shape[0], 1)[0], :]
            curcenter = np.mean(point_set, axis=0)
            random_offset = np.random.uniform(-0.2, 0.2, size=(3,))
            curcenter = curcenter+random_offset
            curmin = curcenter - [space_l/2, space_w/2, space_h/2]
            curmax = curcenter + [space_l/2, space_w/2, space_h/2]
            curmin[2] = coordmin[2] - 0.2
            curmax[2] = coordmax[2] + 0.2
            curchoice = np.sum((point_set >= (curmin - 0.2)) *
                               (point_set <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_segment_set = np.array(semantic_seg)[curchoice]
            if cur_point_set.shape[0] == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.01)) *
                          (cur_point_set <= (curmax + 0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :] - curmin) /
                           (curmax - curmin) * [62.0, 62.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 62.0 * 62.0 +
                             vidx[:, 1] * 62.0 + vidx[:, 2])
            # np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and
            isvalid = len(vidx) / 62.0 / 62.0 / 62.0 >= 0.02
            if isvalid:
                break

        choice = np.random.choice(
            cur_point_set.shape[0], self.npoints, replace=True)
        point_set = cur_point_set[choice, :]
        new_label = cur_segment_set[choice]
        return point_set, new_label, self.filename_list[index]

    def __len__(self):
        return len(self.scene_points_list)
