#!/bin/bash

mkdir build
cd build

BUILD_CONFIG=Release

# choose different screen settings for OS X and Linux
if [ `uname` = "Darwin" ]; then
    SCREEN_ARGS=(
        "-DVTK_USE_X:BOOL=OFF"
        "-DVTK_USE_COCOA:BOOL=ON"
        "-DVTK_USE_CARBON:BOOL=OFF"
    )
else
    SCREEN_ARGS=(
        "-DVTK_USE_X:BOOL=ON"
    )
fi

cmake .. -G "Ninja" \
    -Wno-dev \
    -DCMAKE_BUILD_TYPE=$BUILD_CONFIG \
    -DCMAKE_PREFIX_PATH:PATH="${PREFIX}" \
    -DCMAKE_INSTALL_PREFIX:PATH="${PREFIX}" \
    -DCMAKE_INSTALL_RPATH:PATH="${PREFIX}/lib" \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DVTK_WRAP_PYTHON:BOOL=ON \
    -DVTK_MODULE_ENABLE_VTK_IOGDAL:BOOL=WANT \
    -DVTK_MODULE_ENABLE_VTK_IOPDAL:BOOL=WANT \
    -DVTK_PYTHON_VERSION:STRING="3" \
    -DPython3_EXECUTABLE:PATH="${PREFIX}/bin/python" \
    -DVTK_HAS_FEENABLEEXCEPT:BOOL=OFF \
    ${SCREEN_ARGS[@]}

# compile & install!
ninja install
