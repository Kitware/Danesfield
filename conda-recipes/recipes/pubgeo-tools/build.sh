#!/bin/bash

mkdir build
cd build

BUILD_CONFIG=Release

cmake -G "Ninja" \
    -DCMAKE_BUILD_TYPE=$BUILD_CONFIG \
    -DCMAKE_PREFIX_PATH:PATH="${PREFIX}" \
    -DCMAKE_INSTALL_PREFIX:PATH="${PREFIX}" \
    -DPUBGEO_INSTALL_ALIGN3D:BOOL=ON \
    -DPUBGEO_INSTALL_SHR3D:BOOL=ON \
    ..

ninja install
