Danesfield Demo Instructions
============================

The Danesfield system computes 3D building models from high resolution satellite imagery or geospatial point cloud sources.
The `demonstration script <danesfield_demo.py>`_ provides an example of how to run this software on a publicly
accessible `dataset of multi-view satellite imagery <https://spacenet.ai/core3d/>`_.
The software has a complex set of dependencies, so this demo uses a pre-built Docker containing everthing that is needed.
The demo script makes it easy to pull down required Docker container, imagery data, and model files and run the algorithms.
This can be done in multiple steps or all in one command.
The final output of the system is a collection of 3D building meshes encoded as 3D Tiles, which can be viewed in a web browser using Cesium.JS.

Note that the software does rely on GPU acceleration using CUDA in a few steps, so Nvidia-Docker is required as well as a
modern Nvidia GPU.
In particular we use `VisSat Satellite Stereo <https://github.com/Kai-46/VisSatSatelliteStereo>`_, a third-party tool for computing point clouds from multi-view satellite images.  
VisSat is included in the Danesfield container.  
VisSat can be quite GPU intensive for larger scenes and may require a GPU with more than 8GB of VRAM.  

Software Requirements:
----------------------

- Docker - https://www.docker.com/
- Nvidia-Docker - https://github.com/NVIDIA/nvidia-docker
- Python3 - https://www.python.org/
- Boto3 python package - Boto3 Quickstart

The boto3 and configparser packages can be installed with the following command:
::
	pip install boto3 configparser

Downloading the Danesfield Model Files
--------------------------------------

The learned model files for Danesfield's AI components are hosted separately on data.kitware.com. In the future these should be added to the Docker container. For now they require a separate download. This is achieved with the ``--pull_models`` option. These models are extracted into the Models directory unless overridden with the ``--model_path`` option. 

Downloading CORE3D Image Data
-----------------------------

There are three sites (Jacksonville, UCSD, Omaha) currently available on IARPA's `CORE3D dataset <https://spacenet.ai/core3d/>`_ with WorldView3 imagery. To download the data for a site, provide the site's name using the ``--site`` argument. You will also need to specify a location to save the images using the ``--img_path`` argument. The save directory does not have to already exist, and if one is not provided, your current working directory will be chosen as a default. 

Valid AWS credentials will need to be passed to the script to access the CORE3D dataset. To do this, use the ``--key_id`` and ``--secret_key`` arguments. It's possible that more sites will be added to the dataset in the future, so to view all available sites, use the ``--show-sites`` option. 

Configuring Danesfield
----------------------

Two configuration files are needed for Danesfield. The first is required to be named ``input_<region>.ini``, where <region> is replaced by the name of the site. An example of what information it needs can be found at https://github.com/Kitware/Danesfield/blob/master/input.ini.
The only fields that need to be filled out in this file are ``gsd`` and ``cuda``. The first is the desired ground sample distance (default .25), and the second is whether or not CUDA is available. If Danesfield is being run without images, the gsd field can be ignored. 
The other configuration file is for VisSat Satellite Stereo, and it must be named ``<region>_config.json``. An example of this file can be found at https://github.com/Kitware/Danesfield/blob/master/tools/example_vissat_config.json. The ``bounding_box`` fields are related to the area that you want to reconstruct with Danesfield. They include the UTM zone, easting, and northing of the bounding box's upper left corner, as well as its width and height in meters. The ``steps_to_run`` fields should be as they appear in the example. The final fields, ``alt_min`` and ``alt_max``, are the approximate minimum and maximum altitudes present in the area of interest. This file is not necessary if Danesfield is being run without images.

The two example configuration files linked above should be copied into the configuration directory and renamed to the required names. Edit the necessary fields and the rest of the fields will be filled in by the demo script. 

Three example configuration files, one for each available region, come with this demo. 

Running Danesfield
------------------

After downloading the Docker container, imagery, and models, the demo script can run the Danesfield container on the source imagery. This is achieved with the ``--run`` option. When running Danesfield, one must specify the location of the source imagery, model files, an output directory, and configuration files. Use the ``--model_path`` argument to provide the directory where the necessary model files are located. Use the ``--config_path`` argument to provide the directory where the necessary configuration files are located. Finally, use the ``--out_path`` argument to provide the directory where Danesfield should save all its output. If none of these paths are specified, then these directories will be created within the current directory as a default. 

The full docker run command executed by this demo script will look something like this:
::
	docker run -it --gpus all --shm-size 8G -v /home/danesfield/Demo/imgpath:/mnt \
	-v /home/danesfield/Demo/outdir:/workdir -v /home/danesfield/Demo/configdir:/configs \
	-v /home/danesfield/Demo/Models:/models kitware/danesfield source \
	/opt/conda/etc/profile.d/conda.sh && conda activate core3d && \
	python danesfield/tools/run_danesfield.py --image --vissat --roads \
	/configs/input_Jacksonville.ini

This can be used as a starting point for manually running the Danesfield Docker container in other ways. There are two main ways to run the Danesfield pipeline: with and without images. The ``--image`` option along with the ``--vissat`` option indicate that the pipeline should start with satellite images and generate a point cloud using VisSat. VisSat will save the final point cloud to the path specified under the ``p3d_path`` field in the Danesfield configuration file (see section above). No point cloud needs to be provided by the user in this case. If the demo script is being used to run Danesfield, ``p3d_path`` will automatically be set to ``<out_path>/<site>/cloud.las``. To process an existing point cloud instead of generating one from images, omit both the ``--vissat`` and ``--image`` options. Now the ``p3d_path`` field in the configuration file must be set by the user to the starting point cloud's location. This can also be accomplished by using the ``--point_cloud`` option on the demo script. Use that option to provide the location of the starting point cloud, and the script will then copy that point cloud into the specified output directory and automatically fill in ``p3d_path`` with that new location. If this option is used, the ``--site`` argument should still be used to provide a site name as it will be used as a prefix for naming output files and directories. No images will be downloaded in this case because they are not needed.

There is a third option for running Danesfield's container from the command line. Using only the ``--image`` option without the ``--vissat`` option will allow the user to start with an existing point cloud (the omission of ``--vissat`` means a cloud must be provided), but still run some of the intermediate steps that require images. Notably, the image orthorectification, material classification, and texturing steps can now run with the provided images. These images will not, however, contribute to the point cloud or the reconstruction mesh in any way.

Other Notes
-----------

Danesfield's final tiled results can be found in the ``tiler`` folder in the output directory. If images were used, textured meshes can be found in the ``texture-mapping`` folder in the output directory, and if no images were used, then meshes without texture can be found in the ``roof-geon-extraction`` folder. 

To check for a more recent release of Danesfield's Docker image, use the ``--pull_image`` option. 

Run with ``--help`` to se a quick explanation of each command line argument. 

To get a shell in the container without having to run Danesfield's pipeline, run 
::
	docker exec -it kitware/danesfield /bin/bash

All example configuration files were made with the assumption that user-specified directories were mounted to the Danesfield Docker container as they are in the example run command in the 'Running Danesfield' section. For instance, the user-specified ``imgpath`` becomes ``/mnt`` in the container. Users should change the configuration files to reflect their own mount locations if they choose to run Danesfield without the demo script. 

Visualizing Results
-------------------

The final 3D tiles outputted by Danesfield can be visualized in a web browser using Cesium.JS.

- Install Python 3
- Copy the ``demo/index.html`` file from this repository into the ``tiler`` directory containing the results you want to visualize
- In your terminal, navigate to that ``tiler`` directory and start an HTTP server by running ``python3 -m http.server``
- Go to ``http://localhost:8000/`` in your browser
