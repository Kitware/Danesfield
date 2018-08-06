##Columbia Geon Segmentation Model for Core3D

It takes building point cloud as input and output label file contains the building roof type label for each point. 0: flat roof, 1: slope roof, 2:cylinder roof, 3: sphere roof 

The model and test files are in data.kitware

[https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5b68a3fe8d777f06857c1f24](https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5b68a3fe8d777f06857c1f24)

Compile:

Change the path in the script first before you compiling the file.

```

cd ./tf_ops/grouping/
./tf_grouping_compile.sh
cd ../
cd ./tf_ops/interpolation/
./tf_interpolate_compile.sh
cd ../
cd ./tf_ops/sampling/
./tf_sampling_compile.sh
cd ../

```

Test:

```
cd ./tensorflow/
python roof_segmentation.py 
--model=../model/dayton_geon
--input_pc=/home/xuzhang/project/Core3D/core3d-columbia/data/D2_mls_building.txt
--output_png=../segmentation_graph/out.png
--output_txt=../outlas/out.txt
--text_output

```
