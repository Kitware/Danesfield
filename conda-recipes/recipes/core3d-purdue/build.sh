
#!/usr/bin/env bash

cd Segmentation/code
mkdir build
cd build
# Make libraries
cmake -G "Ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=${PREFIX} \
      -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} \
      -DEigen3_INCLUDE_DIR:PATH=${PREFIX}\include\eigen3 \
      ..
ninja install

cd ../../..
cd Reconstruction/code
mkdir build
cd build
# Make libraries
cmake -G "Ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH:PATH="${PREFIX}" \
      -DCMAKE_INSTALL_PREFIX:PATH="${PREFIX}" \
      ..
ninja install

