Danesfield Demo Instructions
============================

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

Running Danesfield
------------------

After downloading the Docker container, imagery, and models, the script can run the Danesfield container on the source imagery. This is achieved with the ``--run`` option. When running Danesfield, one must specify the location of the source imagery, model files, an output directory, and configuration files. Use the ``--model_path`` argument to provide the directory where the necessary model files are located. Use the ``--config_path`` argument to provide the directory where the necessary configuration files are located. Finally, use the ``--out_path`` argument to provide where the directory where Danesfield should save all its output.

The full docker run command executed by this demo script will look something like this:
::
	docker run -it --gpus all --shm-size 8G -v /home/danesfield/Demo/imgpath:/mnt -v /home/danesfield/Demo/outdir:/workdir -v /home/danesfield/Demo/configdir:/configs -v /home/danesfield/Demo/Models:/models kitware/danesfield source /opt/conda/etc/profile.d/conda.sh && conda activate core3d && python danesfield/tools/run_danesfield.py --image --vissat /configs/input_Jacksonville.ini.

This can be used as a starting point for manually running the Danesfield Docker container in other ways. Notably, the ``--image`` option indicates that the pipeline should start with satellite images (instead of a point cloud) and the ``--vissat`` option indicates that VisSat should be used to generate a point cloud from the images. To process a point cloud, leave off both of these options and in the Danesfield configuration (see section below) set ``p3d_path`` to the point cloud file to use. 

Configuring Danesfield
----------------------

Two configuration files are needed for Danesfield. The first is required to be named ``input_<region>.ini``, where <region> is replaced by the name of teh site. An example of what information it needs can be found at https://github.com/Kitware/Danesfield/blob/master/tools/example_vissat_config.json.
The only fields that need to be filled out in this file are ``gsd`` and ``cuda``. The first is the desired ground sample distance (default .25), and the second is whether or not Cuda is available. 
The other configuration file is for VisSat Satellite Stereo, and it must be named ``<region>_config.json``. An example of this file can be found at https://github.com/Kitware/Danesfield/blob/master/input.ini. The ``bounding_box`` fields are related to the area that you want to reconstruct with Danesfield. They include the UTM zone, easting, and northing of the bounding box's upper left corner, as well as its width and height in meters. The ``steps_to_run`` fields should be as they appear in the example. The final fields, ``alt_min`` and ``alt_max``, are the approximate minimum and maximum altitudes present in the area of interest. 

The two example configuration files linked above should be copied into the configuration directory and renamed to the required names. Edit the necessary fields and the rest of the fields will be filled in by the demo script. 

Three example configuration files, one for each available region, come with this demo. 

Other Notes
-----------

To check for a more recent release of Danesfield's Docker image, use the ``--pull_image`` option. 

Run with ``--help`` to se a quick explanation of each command line argument. 

To get a shell in the container without having to run Danesfield's pipeline, run 
::
	docker exec -it kitware/danesfield /bin/bash