##Columbia Geon Segmentation and Fitting for Core3D

It first takes building point cloud as input and output label file contains the building roof type label for each point. 0: flat roof, 1: slope roof, 2:cylinder roof, 3: sphere roof. After that, the curved plane fitting code will take the labelled point cloud as input and fits all curved planes on the scene. The output of this step is the fitting geon file and remaining point cloud for Purdue to further process. 

The model and test files are in data.kitware

https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5b68a3fe8d777f06857c1f24

Compile:

The shared object files needed for this tool are provided by the `core3d-tf_ops` conda package from kitware-danesfield.  However, if you still wish to compile the necessary files ..

Change the path in the script first before you compiling the file.

.. code-block:: bash

    cd ./tf_ops/grouping/

    ./tf_grouping_compile.sh

    cd ../

    cd ./tf_ops/interpolation/

    ./tf_interpolate_compile.sh

    cd ../

    cd ./tf_ops/sampling/

    ./tf_sampling_compile.sh

    cd ../


Test:

.. code-block:: bash

    cd ./tensorflow/

    python roof_segmentation.py \
    --model=../model/dayton_geon \
    --input_pc=/home/xuzhang/project/Core3D/core3d-columbia/data/D1_mls_building.txt \
    --output_png=../segmentation_graph/out.png \
    --output_txt=../outlas/out_D1.txt \
    --text_output

    python fitting_curved_plane.py \
    --input_pc=../outlas/out_D1.txt \
    --output_png=../segmentation_graph/fit_D1.png \
    --output_txt=../outlas/remain_D1.txt \
    --output_geon=../out_geon/D1_Curve_Geon.npy


    python geon_to_mesh.py \
    --input_geon=../out_geon/D1_Curve_Geon.npy \
    --input_dtm=/dvmm-filer2/projects/Core3D/D1_WPAFB/DTMs/D1_DTM.tif \
    --output_mesh=../out_geon/D1_Curve_Mesh.ply'


Also see run_fitting_curve_plane.py and run_geon_to_mesh.py

