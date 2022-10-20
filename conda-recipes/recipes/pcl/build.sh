
#!/usr/bin/env bash
mkdir build
cd build
# Make libraries
cmake -G "Ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=${PREFIX} \
      -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} \
      -DWITH_QT:BOOL=OFF \
      -DWITH_OPENGL:BOOL=OFF \
      -DWITH_VTK:BOOL=OFF \
      -DWITH_QHULL:BOOL=ON \
      ${SRC_DIR}
# compile & install
ninja install

