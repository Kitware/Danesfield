#!/usr/bin/env bash

# Apply a patch to 'pcl_features-1.8.pc', as per issue #97:
# https://github.com/strawlab/python-pcl/issues/97
cat <<'EOF' | patch ${PREFIX}/lib/pkgconfig/pcl_features-1.8.pc
10c10
< Requires:  pcl_common-1.8 pcl_search-1.8 pcl_kdtree-1.8 pcl_octree-1.8 pcl_filters-1.8 pcl_2d-1.8
---
> Requires:  pcl_common-1.8 pcl_search-1.8 pcl_kdtree-1.8 pcl_octree-1.8 pcl_filters-1.8 #pcl_2d-1.8
EOF


exec_prefix=${PREFIX}
libdir=${exec_prefix}/lib
includedir=${PREFIX}/include/c++/v1

cat <<EOF > ${PREFIX}/lib/pkgconfig/clangdev.pc
Name: clangdev
Description: c++ library
Version: 6.0.1
Requires:
Libs: -L${libdir} -lc++
Cflags: -I${includedir}
EOF



# The '--single-version-externally-managed --record=record.txt'
# arguments seem to be necessary to prevent setuptools from
# complaining about missing packages (which are already covered in the
# meta.yaml file)
python setup.py install --single-version-externally-managed --record=record.txt
