###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

""" Manage Generic Point-Cloud Model (GPM) data that is base64 encoded
"""

import base64
import numpy as np
import struct

from scipy.spatial import KDTree

def get_string(pos, data, length=32):
    return (data[pos:pos + length].decode('ascii').rstrip('\x00').strip(),
            pos + length)

def get_uint16(pos, data):
    return int.from_bytes(data[pos:pos + 2], 'little'), pos + 2

def get_uint32(pos, data):
    return int.from_bytes(data[pos:pos + 4], 'little'), pos + 4

def get_int(pos, data):
    return int.from_bytes(data[pos:pos + 4], 'little'), pos + 4

def get_int8(pos, data):
    return int.from_bytes(data[pos:pos + 1], 'little'), pos + 1

def get_double(pos, data):
    return struct.unpack('d', data[pos:pos + 8])[0], pos + 8

def get_double_vec(pos, data):
    retVal = np.zeros(3)
    retPos = pos

    for i in range(3):
        retVal[i] = struct.unpack('d', data[retPos:retPos + 8])[0]
        retPos += 8

    return retVal, retPos

def get_float(pos, data):
    return struct.unpack('f', data[pos:pos + 4])[0], pos + 4

def get_float_vec(pos, data):
    retVal = np.zeros(3)
    retPos = pos

    for i in range(3):
        retVal[i] = struct.unpack('f', data[retPos:retPos + 4])[0]
        retPos += 4

    return retVal, retPos

def get_cov_matrix(pos, data, dim=3):
    retVal = np.zeros((3,3))
    retPos = pos

    for i in range(3):
        retVal[i,i] = struct.unpack('f', data[retPos:retPos + 4])[0]
        retPos += 4

    retVal[0,1] = retVal[1,0] = struct.unpack('f', data[retPos:retPos + 4])[0]
    retPos += 4
    retVal[0,2] = retVal[2,0] = struct.unpack('f', data[retPos:retPos + 4])[0]
    retPos += 4
    retVal[1,2] = retVal[2,1] = struct.unpack('f', data[retPos:retPos + 4])[0]
    retPos += 4

    return retVal, retPos

class GPM(object):
    def __init__(self, metadata):
        """Constructor
        """

        self.metadata = {}

        if 'GPM_Master' in metadata:
            self.metadata['GPM_Master'] = self.load_GPM_Master(
                metadata['GPM_Master'])

        if 'GPM_GndSpace_Direct' in metadata:
            self.metadata['GPM_GndSpace_Direct'] = self.load_GPM_GndSpace_Direct(
                metadata['GPM_GndSpace_Direct'])

        if 'Per_Point_Lookup_Error_Data' in metadata:
            self.metadata['Per_Point_Lookup_Error_Data'] = self.load_Per_Point_Lookup_Error_Data(
                metadata['Per_Point_Lookup_Error_Data'])

        if 'GPM_Unmodeled_Error_Data' in metadata:
            self.metadata['GPM_Unmodeled_Error_Data'] = self.load_GPM_Unmodeled_Error_Data(
                metadata['GPM_Unmodeled_Error_Data'])

        if 'GPM_GndSpace_Direct' in self.metadata:
            self.ap_search = KDTree(self.metadata['GPM_GndSpace_Direct']['AP'])
        else:
            self.ap_search = None

    def load_GPM_Master(self, data):
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0

        retDict['GPM_Version'], currPos = get_string(currPos, ppe_bytes, 10)
        retDict['GPM_Implementation'], currPos = get_string(currPos, ppe_bytes, 20)
        retDict['MCS_ID'], currPos = get_uint16(currPos, ppe_bytes)
        retDict['MCS_ORIGIN_X'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_ORIGIN_Y'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_ORIGIN_Z'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_XUXM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_XUYM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_XUZM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_YUXM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_YUYM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_YUZM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_ZUXM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_ZUYM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_ZUZM'], currPos = get_double(currPos, ppe_bytes)
        retDict['MCS_HEMI'], currPos = get_string(currPos, ppe_bytes, 1)
        retDict['MCS_ZONE'], currPos = get_uint16(currPos, ppe_bytes)

        retDict['DATASET_ID'], currPos = get_string(currPos, ppe_bytes)
        retDict['REFERENCE_DATE_TIME'], currPos = get_string(currPos, ppe_bytes, 18)
        retDict['EX_ORIGIN_X'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_ORIGIN_Y'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_ORIGIN_Z'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_XMUXE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_XMUYE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_XMUZE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_YMUXE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_YMUYE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_YMUZE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_ZMUXE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_ZMUYE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_ZMUZE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_DELTAXE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_DELTAYE'], currPos = get_double(currPos, ppe_bytes)
        retDict['EX_DELTAZE'], currPos = get_double(currPos, ppe_bytes)
        retDict['NUM_COLLECTIONS'], currPos = get_uint32(currPos, ppe_bytes)

        collections = []
        for i in range(retDict['NUM_COLLECTIONS']):
            collect_dict = {}
            collect_dict['COLLECTION_ID'], currPos = get_string(currPos, ppe_bytes)
            collect_dict['PLATFORM_ID'], currPos = get_string(currPos, ppe_bytes)
            collect_dict['NUM_SENSORS'], currPos = get_uint32(currPos, ppe_bytes)

            sensors = []
            for j in range(collect_dict['NUM_SENSORS']):
                sensor_dict = {}
                sensor_dict['SENSOR_ID'], currPos = get_string(currPos, ppe_bytes)
                sensor_dict['SENSOR_TYPE'], currPos = get_string(currPos, ppe_bytes)
                sensor_dict['SENSOR_TYPE'], currPos = get_string(currPos, ppe_bytes)
                sensor_dict['NUM_COLLECTION_UNITS'], currPos = get_uint32(currPos, ppe_bytes)

                collect_units = []
                for k in range(sensor_dict['NUM_COLLECTION_UNITS']):
                    unit_dict = {}
                    unit_dict['REFERENCE_DATE_TIME'], currPos = get_string(currPos, ppe_bytes, 18)
                    unit_dict['COLLECTION_UNIT_ID'], currPos = get_string(currPos, ppe_bytes, 128)
                    unit_dict['POINT_SOURCE_ID'], currPos = get_int(currPos, ppe_bytes)
                    unit_dict['EX_ORIGIN_X'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_ORIGIN_Y'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_ORIGIN_Z'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_XMUXE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_XMUYE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_XMUZE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_YMUXE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_YMUYE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_YMUZE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_ZMUXE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_ZMUYE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_ZMUZE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_DELTAXE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_DELTAYE'], currPos = get_double(currPos, ppe_bytes)
                    unit_dict['EX_DELTAZE'], currPos = get_double(currPos, ppe_bytes)
                    collect_units.append(unit_dict)

                sensor_dict['COLLECTION_UNIT_RECORD'] = collect_units
                sensors.append(sensor_dict)

            collect_dict['SENSOR_RECORD'] = sensors
            collections.append(collect_dict)

        retDict['COLLECTION_RECORD'] = collections

        return retDict

    def load_Per_Point_Lookup_Error_Data(self, data):
        # Decode base64 encoded data
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0
        retDict['NUM_PPE_COV_RECORDS'], currPos = get_uint16(currPos, ppe_bytes)
        retDict['PPE_FIELD_NAME'], currPos = get_string(currPos, ppe_bytes)

        ppe = []
        for n in range(retDict['NUM_PPE_COV_RECORDS']):
            cov_matrix, currPos = get_cov_matrix(currPos, ppe_bytes)
            ppe.append(cov_matrix)

        retDict['PPE_COV_RECORD'] = ppe

        return retDict

    def load_GPM_GndSpace_Direct(self, data):
        # Decode base64 encoded data
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0

        retDict['DATASET_ID'], currPos = get_string(currPos, ppe_bytes)

        # Get 3DC parameter flags as an integer
        param_flags, currPos = get_int8(currPos, ppe_bytes)

        if param_flags:
            retDict['CU_X_COORD_RE_CENTERING_VALUE'], currPos = (
                get_double(currPos, ppe_bytes) )
            retDict['CU_Y_COORD_RE_CENTERING_VALUE'], currPos = (
                get_double(currPos, ppe_bytes) )
            retDict['CU_Z_COORD_RE_CENTERING_VALUE'], currPos = (
                get_double(currPos, ppe_bytes) )
            retDict['CU_S_NORMALIZING_SCALE_FACTOR'], currPos = (
                get_double(currPos, ppe_bytes) )

            # Apply masks to get other 3DC parameters
            if param_flags & 0b00000001:
                retDict['CU_DELTA_X'], currPos = get_double(currPos, ppe_bytes)
            if param_flags & 0b00000010:
                retDict['CU_DELTA_y'], currPos = get_double(currPos, ppe_bytes)
            if param_flags & 0b00000100:
                retDict['CU_DELTA_Z'], currPos = get_double(currPos, ppe_bytes)
            if param_flags & 0b00001000:
                retDict['CU_OMEGA'], currPos = get_double(currPos, ppe_bytes)
            if param_flags & 0b00010000:
                retDict['CU_PHI'], currPos = get_double(currPos, ppe_bytes)
            if param_flags & 0b00100000:
                retDict['CU_KAPPA'], currPos = get_double(currPos, ppe_bytes)
            if param_flags & 0b01000000:
                retDict['CU_DELTA_S'], currPos = get_double(currPos, ppe_bytes)

        retDict['NUM_AP_RECORDS'], currPos = get_uint16(currPos, ppe_bytes)
        retDict['INTERPOLATION_MODE'], currPos = get_uint16(currPos, ppe_bytes)
        retDict['INTERP_NUM_POSTS'], currPos = get_uint16(currPos, ppe_bytes)
        retDict['DAMPENING_PARAM'], currPos = get_double(currPos, ppe_bytes)

        anchorPoints = []
        anchorDeltas = []
        anchorCovar = []

        for n in range(retDict['NUM_AP_RECORDS']):
            ap, currPos = get_double_vec(currPos, ppe_bytes)
            anchorPoints.append(ap)

            ap_delta, currPos = get_float_vec(currPos, ppe_bytes)
            anchorDeltas.append(ap_delta)

        for n in range(retDict['NUM_AP_RECORDS']):
            cov_matrix, currPos = get_cov_matrix(currPos, ppe_bytes)
            anchorCovar.append(cov_matrix)

        retDict['AP'] = anchorPoints
        retDict['AP_DELTA'] = anchorDeltas
        retDict['AP_COVAR'] = anchorCovar

        return retDict

    def load_GPM_Unmodeled_Error_Data(self, data):
        # Decode base64 encoded data
        ppe_bytes = base64.b64decode(data)

        retDict = {}

        currPos = 0

        retDict['NUM_UE_RECORDS'], currPos = get_uint16(currPos, ppe_bytes)

        ue_records = []
        for i in range(retDict['NUM_UE_RECORDS']):
            ue_dict = {}
            ue_dict['TRAJECTORY_ID'], currPos = get_int(currPos, ppe_bytes)
            ue_dict['UNIQUE_ID'], currPos = get_string(currPos, ppe_bytes, length=128)
            ue_dict['CORR_ROT_THETA_X'], currPos = get_double(currPos, ppe_bytes)
            ue_dict['CORR_ROT_THETA_Y'], currPos = get_double(currPos, ppe_bytes)
            ue_dict['CORR_ROT_THETA_Z'], currPos = get_double(currPos, ppe_bytes)
            ue_dict['PARAM_A_U'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_ALPHA_U'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_BETA_U'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_TAU_U'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_A_V'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_ALPHA_V'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_BETA_V'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_TAU_V'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_A_W'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_ALPHA_W'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_BETA_W'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['PARAM_TAU_W'], currPos = get_float(currPos, ppe_bytes)
            ue_dict['NUM_UE_POSTS'], currPos = get_uint16(currPos, ppe_bytes)

            posts = []
            for it in range(ue_dict['NUM_UE_POSTS']):
                post_dict = {}
                post_dict['UE_COV_POST_X'], currPos = get_double(currPos, ppe_bytes)
                post_dict['UE_COV_POST_Y'], currPos = get_double(currPos, ppe_bytes)
                post_dict['UE_COV_POST_Z'], currPos = get_double(currPos, ppe_bytes)
                post_dict['UE_COV_POST_VARX'], currPos = get_float(currPos, ppe_bytes)
                post_dict['UE_COV_POST_VARY'], currPos = get_float(currPos, ppe_bytes)
                post_dict['UE_COV_POST_VARZ'], currPos = get_float(currPos, ppe_bytes)
                post_dict['UE_COV_POST_VARXY'], currPos = get_float(currPos, ppe_bytes)
                post_dict['UE_COV_POST_VARXZ'], currPos = get_float(currPos, ppe_bytes)
                post_dict['UE_COV_POST_VARYZ'], currPos = get_float(currPos, ppe_bytes)

                posts.append(post_dict)

            if posts:
                ue_dict['UE_COV_POST'] = post_dict

            ue_records.append(ue_dict)

        retDict['UE_RECORD'] = ue_records

        return retDict
