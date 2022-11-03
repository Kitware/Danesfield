###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

""" Helper functions and classes for base64 encoding of GPM data
"""

import base64
import json
import numpy as np
import struct

# Methods to decode data from base64 string and move to the next chunk
def to_string(pos, data, length=32):
    return (data[pos:pos + length].decode('ascii').rstrip('\x00').strip(),
            pos + length)

def to_uint16(pos, data):
    return int.from_bytes(data[pos:pos + 2], 'little'), pos + 2

def to_uint32(pos, data):
    return int.from_bytes(data[pos:pos + 4], 'little'), pos + 4

def to_int(pos, data):
    return int.from_bytes(data[pos:pos + 4], 'little'), pos + 4

def to_int8(pos, data):
    return int.from_bytes(data[pos:pos + 1], 'little'), pos + 1

def to_double(pos, data):
    return struct.unpack('d', data[pos:pos + 8])[0], pos + 8

def to_double_vec(pos, data):
    retVal = np.zeros(3)
    retPos = pos

    for i in range(3):
        retVal[i] = struct.unpack('d', data[retPos:retPos + 8])[0]
        retPos += 8

    return retVal, retPos

def to_float(pos, data):
    return struct.unpack('f', data[pos:pos + 4])[0], pos + 4

def to_float_vec(pos, data):
    retVal = np.zeros(3)
    retPos = pos

    for i in range(3):
        retVal[i] = struct.unpack('f', data[retPos:retPos + 4])[0]
        retPos += 4

    return retVal, retPos

def to_cov_matrix(pos, data, dim=3):
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

# Encode numpy array in base64 with key to identify type
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64.decode('utf-8'),
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        super(NumpyArrayEncoder, self).default(obj)

def json_numpy_array_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

