export PREFIX
cd tf_ops/sampling
bash tf_sampling_compile.sh

cd ../grouping
bash tf_grouping_compile.sh

cd ../3d_interpolation
bash tf_interpolate_compile.sh

cd ..
mkdir -p "${PREFIX}"/lib
cp */*.{so,cu.o} "${PREFIX}"/lib/
