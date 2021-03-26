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

def get_double(pos, data):
  return struct.unpack('d', data[pos:pos + 8]), pos + 8

def get_double_vec(pos, data):
  retVal = np.zeros(3)
  retPos = pos

  for i in range(3):
    retVal[i] = struct.unpack('d', data[retPos:retPos + 8])[0]
    retPos += 8

  return retVal, retPos

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
