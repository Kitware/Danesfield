###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

""" Decode Generic Point-Cloud Model (GPM) data that is base64 encoded
"""

import base64
import numpy as np
import struct

def get_unsigned_short(pos, data):
  return int.from_bytes(data[pos:pos + 2], 'little'), pos + 2

def get_string(pos, data):
  return data[pos:pos + 32].decode('ascii').rstrip('\x00'), pos + 32

def get_int(pos, data):
  return int.from_bytes(data[pos:pos + 4], 'little'), pos + 4

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
    retVal[i,i] = struct.unpack(
      'f', data[retPos:retPos + 4])[0]
    retPos += 4

  retVal[0,1] = retVal[1,0] = struct.unpack(
    'f', data[retPos:retPos + 4])[0]
  retPos += 4
  retVal[0,2] = retVal[2,0] = struct.unpack(
    'f', data[retPos:retPos + 4])[0]
  retPos += 4
  retVal[1,2] = retVal[2,1] = struct.unpack(
    'f', data[retPos:retPos + 4])[0]
  retPos += 4

  return retVal, retPos

def load_Per_Point_Lookup_Error_Data(data):
  # Decode base64 encoded data
  ppe_bytes = base64.b64decode(data)

  retDict = {}

  currPos = 0
  retDict['NUM_PPE_COV_RECORDS'], currPos = get_unsigned_short(currPos, ppe_bytes)
  retDict['PPE_FIELD_NAME'], currPos = get_string(currPos, ppe_bytes)

  ppe = []
  for n in range(retDict['NUM_PPE_COV_RECORDS']):
    cov_matrix, currPos = get_cov_matrix(currPos, ppe_bytes)
    ppe.append(cov_matrix)

  retDict['PPE_COV_RECORD'] = ppe

  return retDict

def load_GPM_GndSpace_Direct(data):
  # Decode base64 encoded data
  ppe_bytes = base64.b64decode(data)

  retDict = {}

  currPos = 0

  retDict['DATASET_ID'], currPos = get_string(currPos, ppe_bytes)

  # Skip for now
  currPos += 1 # 3DC_PARAM_INDEX_FLAGS

  retDict['NUM_AP_RECORDS'], currPos = get_unsigned_short(currPos, ppe_bytes)
  retDict['INTERPOLATION_MODE'], currPos = get_unsigned_short(currPos, ppe_bytes)
  retDict['INTERP_NUM_POSTS'], currPos = get_unsigned_short(currPos, ppe_bytes)
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

def load_GPM_Unmodeled_Error_Data(data):
  # Decode base64 encoded data
  ppe_bytes = base64.b64decode(data)

  retDict = {}

  currPos = 0

  retDict['NUM_UE_RECORDS'], currPos = get_unsigned_short(currPos, ppe_bytes)

  ue_records = []

  for i in range(retDict['NUM_UE_RECORDS']):
    ue_dict = {}
    ue_dict['TRAJECTORY_ID'], currPos = get_int(currPos, ppe_bytes)
    ue_dict['UNIQUE_ID'], currPos = get_string(currPos, ppe_bytes)
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

    ue_dict['NUM_UE_POSTS'], currpos = get_unsigned_short(currPos, ppe_bytes)

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
