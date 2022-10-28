#!/usr/bin/env bash

mkdir build
cd build

# Make libraries
cmake -G "Ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=${PREFIX} \
      -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} \
      -DBUILD_MATLAB_BINDINGS:BOOL=OFF \
      -DBUILD_TESTS:BOOL=OFF \
      -DBUILD_EXAMPLES:BOOL=OFF \
      ${SRC_DIR}

# compile and install
ninja install

