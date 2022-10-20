#!/bin/bash

mkdir build
cd build

BUILD_CONFIG=Release

if [ $(uname) == Darwin ]; then
    export CXXFLAGS="${CXXFLAGS} -stdlib=libc++"
    export LDFLAGS="${LDFLAGS} -headerpad_max_install_names"
fi



cmake ../Texture_Atlas/src/ -G "Ninja" \
    -Wno-dev \
    -DCMAKE_BUILD_TYPE=$BUILD_CONFIG \
    -DCMAKE_PREFIX_PATH:PATH="${PREFIX}" \
    -DCMAKE_INSTALL_PREFIX:PATH="${PREFIX}" \
    -DCMAKE_INSTALL_RPATH:PATH="${PREFIX}/lib" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING="10.9"

# compile & install!
ninja install
