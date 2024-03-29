###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

""" Manage Generic Point-Cloud Model (GPM) data that is base64 encoded
"""

import base64
import danesfield.gpm_decode64 as gpm64
import json
import numpy as np
import struct

from scipy.spatial import KDTree

# Recursively search json GPM metadata
def search_json(key, json_data, matches):
    if type(json_data) is dict:
        for k in json_data:
            if k == key:
                matches.append(json_data[k])
            elif (k.startswith('vlr_') and
                  type(json_data[k]) is dict and
                  json_data[k]['description'] == key):
                matches.append(json_data[k]['data'])
            else:
                search_json(key, json_data[k], matches)
    elif type(json_data) is list:
        for l in json_data:
            search_json(key, l, matches)

class GPM(object):
    def __init__(self, metadata):
        """Constructor
        """
        self.metadata = {}

        # The number of 3DC parameters
        self.num_3DC = 0

        # Check if metadata is from pdal
        if 'metadata' in metadata:
            self.loadFromPdal(metadata['metadata'])
        # If not assume we are loading from an export
        else:
            self.loadFromExport(metadata)

        if not self.metadata:
            print('WARNING: No GPM metadata found.')

        # Create the anchor point search tree
        if 'GPM_GndSpace_Direct' in self.metadata:
            self.ap_tree = KDTree(self.metadata['GPM_GndSpace_Direct']['AP'])
        else:
            self.ap_tree = None

        # Set up anchor point interpolation
        if 'GPM_GndSpace_Direct' in self.metadata:
            self.interpolation_type = (
                  self.metadata['GPM_GndSpace_Direct']['INTERPOLATION_MODE']
            )
            if self.interpolation_type == 0:
                self.num_interpolate_ap = 8
            else:
                self.num_interpolate_ap = (
                    self.metadata['GPM_GndSpace_Direct']['INTERP_NUM_POSTS']
                )
                self.D = self.metadata['GPM_GndSpace_Direct']['DAMPENING_PARAM']

        # Set up unmodeled error search trees
        if 'GPM_Unmodeled_Error_Data' in self.metadata:
            self.ue_trees = []
            ue_metadata = self.metadata['GPM_Unmodeled_Error_Data']
            for i in range(ue_metadata['NUM_UE_RECORDS']):
                self.ue_trees.append(KDTree(ue_metadata['UE_RECORD'][i]['UE_PTS']))

    # Populate GPM metadata from pdal metadata
    def loadFromPdal(self, metadata):
        # Search for the possible GPM metadata
        matches = []
        search_json('GPM_Master', metadata, matches)
        if matches:
            self.metadata['GPM_Master'] = self.load_GPM_Master(matches[0])

        matches = []
        search_json('GPM_GndSpace_Direct', metadata, matches)
        if matches:
            self.metadata['GPM_GndSpace_Direct'] = (
                  self.load_GPM_GndSpace_Direct(matches[0])
            )

        matches = []
        search_json('Per_Point_Lookup_Error_Data', metadata, matches)
        if matches:
            self.metadata['Per_Point_Lookup_Error_Data'] = (
                  self.load_Per_Point_Lookup_Error_Data(matches[0])
            )

        matches = []
        search_json('GPM_Unmodeled_Error_Data', metadata, matches)
        if matches:
            self.metadata['GPM_Unmodeled_Error_Data'] = (
                  self.load_GPM_Unmodeled_Error_Data(matches[0])
            )

    def loadFromExport(self, metadata):
        for k in metadata:
            if k in {'GPM_Master', 'GPM_GndSpace_Direct',
                     'Per_Point_Lookup_Error_Data', 'GPM_Unmodeled_Error_Data'}:
                self.metadata[k] = metadata[k]

    # Set up PPE lookup if that data is available
    def setupPPELookup(self, points, indices):
        self.ppe_tree = KDTree(points)
        self.ppe_indices = indices.astype(int)

    def checkBytesProcessed(self, endPos, data, name):
        if (endPos != len(data)):
            print('WARNING: last byte position: ', endPos, ' does not match'
                  ' number of bytes: ', len(data), ' for ', name)

    def get_weights(self, dist):
        weights = np.zeros(dist.shape)
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                weights[i, j] = np.exp(-dist[i,j]/self.D)
        denom = np.sum(weights, axis=1).reshape( (dist.shape[0], 1) )

        return np.divide(weights, denom)

    def get_covar(self, points):
        distances, indices  = (
            self.ap_tree.query(points, k=self.num_interpolate_ap)
        )
        wts = self.get_weights(distances)
        covar = np.diagonal(
            self.metadata['GPM_GndSpace_Direct']['COVAR_AP']).transpose()
        ap_covar = covar[indices]
        return np.sum(wts.reshape(wts.shape + (1,1))*ap_covar, axis=1)

    def get_per_point_error(self, points):
        # Nearest neighbor interpolation
        distances, ap_indices = (
            self.ppe_tree.query(points, k=1)
        )
        covar = (
            self.metadata['Per_Point_Lookup_Error_Data']['PPE_COV_RECORD']
        )
        return covar[self.ppe_indices[ap_indices]]

    def get_unmodeled_error(self, points):
        # Nearest neighbor interpolation
        distances, ue_indices = (
            self.ue_trees[0].query(points, k=1)
        )
        covar = (
            self.metadata['GPM_Unmodeled_Error_Data']['UE_RECORD'][0]['UE_COV']
        )
        return covar[ue_indices]

    def load_GPM_Master(self, data):
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0

        retDict['GPM_Version'], currPos = gpm64.to_string(currPos, ppe_bytes, 10)
        retDict['GPM_Implementation'], currPos = gpm64.to_string(currPos, ppe_bytes, 20)
        retDict['MCS_ID'], currPos = gpm64.to_uint16(currPos, ppe_bytes)
        retDict['MCS_ORIGIN_X'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_ORIGIN_Y'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_ORIGIN_Z'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_XUXM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_XUYM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_XUZM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_YUXM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_YUYM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_YUZM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_ZUXM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_ZUYM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_ZUZM'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['MCS_HEMI'], currPos = gpm64.to_string(currPos, ppe_bytes, 1)
        retDict['MCS_ZONE'], currPos = gpm64.to_uint16(currPos, ppe_bytes)

        retDict['DATASET_ID'], currPos = gpm64.to_string(currPos, ppe_bytes)
        retDict['REFERENCE_DATE_TIME'], currPos = gpm64.to_string(currPos, ppe_bytes, 18)
        retDict['EX_ORIGIN_X'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_ORIGIN_Y'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_ORIGIN_Z'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_XMUXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_XMUYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_XMUZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_YMUXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_YMUYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_YMUZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_ZMUXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_ZMUYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_ZMUZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_DELTAXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_DELTAYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['EX_DELTAZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
        retDict['NUM_COLLECTIONS'], currPos = gpm64.to_uint32(currPos, ppe_bytes)

        collections = []
        for i in range(retDict['NUM_COLLECTIONS']):
            collect_dict = {}
            collect_dict['COLLECTION_ID'], currPos = gpm64.to_string(currPos, ppe_bytes)
            collect_dict['PLATFORM_ID'], currPos = gpm64.to_string(currPos, ppe_bytes)
            collect_dict['NUM_SENSORS'], currPos = gpm64.to_uint32(currPos, ppe_bytes)

            sensors = []
            for j in range(collect_dict['NUM_SENSORS']):
                sensor_dict = {}
                sensor_dict['SENSOR_ID'], currPos = gpm64.to_string(currPos, ppe_bytes)
                sensor_dict['SENSOR_TYPE'], currPos = gpm64.to_string(currPos, ppe_bytes)
                sensor_dict['SENSOR_TYPE'], currPos = gpm64.to_string(currPos, ppe_bytes)
                sensor_dict['NUM_COLLECTION_UNITS'], currPos = gpm64.to_uint32(currPos, ppe_bytes)

                collect_units = []
                for k in range(sensor_dict['NUM_COLLECTION_UNITS']):
                    unit_dict = {}
                    unit_dict['REFERENCE_DATE_TIME'], currPos = gpm64.to_string(currPos, ppe_bytes, 18)
                    unit_dict['COLLECTION_UNIT_ID'], currPos = gpm64.to_string(currPos, ppe_bytes, 128)
                    unit_dict['POINT_SOURCE_ID'], currPos = gpm64.to_int(currPos, ppe_bytes)
                    unit_dict['EX_ORIGIN_X'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_ORIGIN_Y'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_ORIGIN_Z'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_XMUXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_XMUYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_XMUZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_YMUXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_YMUYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_YMUZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_ZMUXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_ZMUYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_ZMUZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_DELTAXE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_DELTAYE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    unit_dict['EX_DELTAZE'], currPos = gpm64.to_double(currPos, ppe_bytes)
                    collect_units.append(unit_dict)

                sensor_dict['COLLECTION_UNIT_RECORD'] = collect_units
                sensors.append(sensor_dict)

            collect_dict['SENSOR_RECORD'] = sensors
            collections.append(collect_dict)

        retDict['COLLECTION_RECORD'] = collections

        self.checkBytesProcessed(currPos, ppe_bytes, 'GPM_Master')

        return retDict

    def load_Per_Point_Lookup_Error_Data(self, data):
        # Decode base64 encoded data
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0
        retDict['NUM_PPE_COV_RECORDS'], currPos = gpm64.to_uint16(currPos, ppe_bytes)
        retDict['PPE_FIELD_NAME'], currPos = gpm64.to_string(currPos, ppe_bytes)

        ppe = np.zeros((retDict['NUM_PPE_COV_RECORDS'], 3, 3))
        for i in range(retDict['NUM_PPE_COV_RECORDS']):
            cov_matrix, currPos = gpm64.to_cov_matrix(currPos, ppe_bytes)
            ppe[i, :, :] = cov_matrix

        retDict['PPE_COV_RECORD'] = ppe

        self.checkBytesProcessed(currPos, ppe_bytes, 'Per_Point_Lookup_Error_Data')

        return retDict

    def load_GPM_GndSpace_Direct(self, data):
        # Decode base64 encoded data
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0

        retDict['DATASET_ID'], currPos = gpm64.to_string(currPos, ppe_bytes)

        # Get 3DC parameter flags as an integer
        param_flags, currPos = gpm64.to_int8(currPos, ppe_bytes)

        if param_flags:
            retDict['CU_X_COORD_RE_CENTERING_VALUE'], currPos = (
                gpm64.to_double(currPos, ppe_bytes) )
            retDict['CU_Y_COORD_RE_CENTERING_VALUE'], currPos = (
                gpm64.to_double(currPos, ppe_bytes) )
            retDict['CU_Z_COORD_RE_CENTERING_VALUE'], currPos = (
                gpm64.to_double(currPos, ppe_bytes) )
            retDict['CU_S_NORMALIZING_SCALE_FACTOR'], currPos = (
                gpm64.to_double(currPos, ppe_bytes) )

            # Apply masks to get other 3DC parameters
            if param_flags & 0b00000001:
                retDict['CU_DELTA_X'], currPos = gpm64.to_double(currPos, ppe_bytes)
                self.num_3DC += 1
            else:
                retDict['CU_DELTA_X'] = 0
            if param_flags & 0b00000010:
                retDict['CU_DELTA_Y'], currPos = gpm64.to_double(currPos, ppe_bytes)
                self.num_3DC += 1
            else:
                retDict['CU_DELTA_Y'] = 0
            if param_flags & 0b00000100:
                retDict['CU_DELTA_Z'], currPos = gpm64.to_double(currPos, ppe_bytes)
                self.num_3DC += 1
            else:
                retDict['CU_DELTA_Z'] = 0
            if param_flags & 0b00001000:
                retDict['CU_OMEGA'], currPos = gpm64.to_double(currPos, ppe_bytes)
                self.num_3DC += 1
            else:
                retDict['CU_OMEGA'] = 0
            if param_flags & 0b00010000:
                retDict['CU_PHI'], currPos = gpm64.to_double(currPos, ppe_bytes)
            if param_flags & 0b00100000:
                retDict['CU_KAPPA'], currPos = gpm64.to_double(currPos, ppe_bytes)
                self.num_3DC += 1
            else:
                retDict['CU_KAPPA'] = 0
            if param_flags & 0b01000000:
                retDict['CU_DELTA_S'], currPos = gpm64.to_double(currPos, ppe_bytes)
                self.num_3DC += 1
            else:
                retDict['CU_DELTA_S'] = 0

        retDict['NUM_AP_RECORDS'], currPos = gpm64.to_uint16(currPos, ppe_bytes)
        retDict['INTERPOLATION_MODE'], currPos = gpm64.to_uint16(currPos, ppe_bytes)
        retDict['INTERP_NUM_POSTS'], currPos = gpm64.to_uint16(currPos, ppe_bytes)
        retDict['DAMPENING_PARAM'], currPos = gpm64.to_double(currPos, ppe_bytes)

        anchorPoints = np.zeros((retDict['NUM_AP_RECORDS'], 3))
        anchorDeltas = np.zeros((retDict['NUM_AP_RECORDS'], 3))

        if self.num_3DC > 0:
            covar_3DC = np.zeros((self.num_3DC, self.num_3DC))
            covar_3DC_AP = np.zeros((self.num_3DC, retDict['NUM_AP_RECORDS'], 3))
        covar_AP = np.zeros((retDict['NUM_AP_RECORDS'],
                             retDict['NUM_AP_RECORDS'],
                             3, 3))

        for i in range(retDict['NUM_AP_RECORDS']):
            ap, currPos = gpm64.to_double_vec(currPos, ppe_bytes)
            anchorPoints[i,:] = ap

            ap_delta, currPos = gpm64.to_float_vec(currPos, ppe_bytes)
            anchorDeltas[i,:] = ap_delta

        for i in range(self.num_3DC):
            for j in range(i + 1):
                elem, currPos = gpm64.to_float(currPos, ppe_bytes)
                covar_3DC[i, j] = elem
                covar_3DC[j, i] = elem

        # Full cross covariance matrix stored in column major order
        for cj in range(3*retDict['NUM_AP_RECORDS']):
            c = cj//3
            j = cj%3
            for r in range(self.num_3DC):
                covar_3DC_AP[r, c, j], currPos = gpm64.to_float(currPos, ppe_bytes)

            for ri in range(cj+1):
                r = ri//3
                i = ri%3
                elem, currPos = gpm64.to_float(currPos, ppe_bytes)
                covar_AP[r, c, i, j] = elem
                covar_AP[c, r, j, i] = elem

        retDict['AP'] = anchorPoints
        retDict['AP_DELTA'] = anchorDeltas

        if self.num_3DC > 0:
            retDict['COVAR_3DC'] = covar_3DC
            retDict['COVAR_3DC_AP'] = covar_3DC_AP

        retDict['COVAR_AP'] = covar_AP

        self.checkBytesProcessed(currPos, ppe_bytes, 'GPM_GndSpace_Direct')

        return retDict

    def load_GPM_Unmodeled_Error_Data(self, data):
        # Decode base64 encoded data
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0

        retDict['NUM_UE_RECORDS'], currPos = gpm64.to_uint16(currPos, ppe_bytes)

        ue_records = []
        for i in range(retDict['NUM_UE_RECORDS']):
            ue_dict = {}
            ue_dict['TRAJECTORY_ID'], currPos = gpm64.to_int(currPos, ppe_bytes)
            ue_dict['UNIQUE_ID'], currPos = gpm64.to_string(currPos, ppe_bytes, length=128)
            ue_dict['CORR_ROT_THETA_X'], currPos = gpm64.to_double(currPos, ppe_bytes)
            ue_dict['CORR_ROT_THETA_Y'], currPos = gpm64.to_double(currPos, ppe_bytes)
            ue_dict['CORR_ROT_THETA_Z'], currPos = gpm64.to_double(currPos, ppe_bytes)
            ue_dict['PARAM_A_U'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_ALPHA_U'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_BETA_U'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_TAU_U'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_A_V'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_ALPHA_V'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_BETA_V'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_TAU_V'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_A_W'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_ALPHA_W'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_BETA_W'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['PARAM_TAU_W'], currPos = gpm64.to_float(currPos, ppe_bytes)
            ue_dict['NUM_UE_POSTS'], currPos = gpm64.to_uint16(currPos, ppe_bytes)

            uePoints = np.zeros((ue_dict['NUM_UE_POSTS'], 3))
            ueCovar = np.zeros((ue_dict['NUM_UE_POSTS'], 3, 3))
            for it in range(ue_dict['NUM_UE_POSTS']):
                for j in range(3):
                    ptVal, currPos = gpm64.to_double(currPos, ppe_bytes)
                    uePoints[it, j] = ptVal
                for j in range(3):
                    covarDiag, currPos = gpm64.to_float(currPos, ppe_bytes)
                    ueCovar[it, j, j] = covarDiag
                covarOffdiag, currPos = gpm64.to_float_vec(currPos, ppe_bytes)
                ueCovar[it, 0, 1] = ueCovar[it, 1, 0] = covarOffdiag[0]
                ueCovar[it, 0, 2] = ueCovar[it, 2, 0] = covarOffdiag[1]
                ueCovar[it, 1, 2] = ueCovar[it, 2, 1] = covarOffdiag[2]

            ue_dict['UE_PTS'] = uePoints
            ue_dict['UE_COV'] = ueCovar

            ue_records.append(ue_dict)

        retDict['UE_RECORD'] = ue_records

        self.checkBytesProcessed(currPos, ppe_bytes, 'GPM_Unmodeled_Error_Data')

        return retDict
