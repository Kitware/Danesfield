==========
Danesfield
==========

This repository addresses the algorithmic challenges of the IARPA CORE3D
program.  The goal of this software is to reconstruct semantically meaningful
3D models of buildings and other man-made structures from satellite imagery.

.. image:: danesfield_system_graphic.png
    :align: center
    :alt: The Danesfield system performs 3D reconstruction from satellite imagery

This software is designed to process multiple view satellite imagery and is
currently configured to process collections of panchromatic and multispectral
WorldView3 imagery such as the examples provided in the public
`CORE3D Dataset <https://spacenet.ai/core3d/>`_.
The algorithms were described in a **Best Paper** awarding winning paper at
EarthVision_ 2019:

    M. Leotta, C. Long, B. Jacquet, M. Zins, D. Lipsa, J. Shan, B. Xu, Z. Li,
    X. Zhang, S. Chang, M. Purri, J. Xue, and K. Dana,
    "`Urban Semantic 3D Reconstruction From Multiview Satellite Imagery`__,"
    in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
    Workshops: EarthVision, 2019.

The Danesfield project has evolved since this publication.
For details, see `Project History`_.

This repository contains the algorithms to solve the CORE3D problem, but a
web-based user interface and cloud-based processing infrastructure are provided
in a separate project called
`Danesfield-App <https://github.com/Kitware/Danesfield-App>`_.
The algorithms in this repository
are written in Python or provide a Python interface.


Getting Started
===============

Clone repository
----------------

Clone this repository with its sub-modules by running:

.. code-block::

    git clone --recursive git@github.com:Kitware/Danesfield.git

To fetch the latest version of this repository and its sub-modules, run:

.. code-block::

    git pull
    git submodule update --init --recursive

Create Conda environment
------------------------

The first step in running or developing Danesfield code is to obtain the
correct development environment.  The Danesfield algorithms require a number of
dependencies on geospatial and computer vision libraries.  Provided with this
repository are instructions for configuring a development environment with
Conda.  Conda provides a consistent development environment with a known
configuration of dependencies versions.  Follow the directions in
`<deployment/conda/README.rst>`_ to setup this environment.

Docker image
------------

This repository has also been built into a Docker image, which
includes the required conda environment.  The image is available from
Docker Hub at `<https://hub.docker.com/r/kitware/danesfield>`_ and can
be pulled down by running ``docker pull kitware/danesfield``.  The
image was built using the Dockerfile included in this repository.

As some of the Danesfield algorithms require a GPU, you'll need to
have `NVIDIA Docker <https://github.com/NVIDIA/nvidia-docker>`_
installed, and use the ``nvidia-docker`` command when running the
image.

Project Layout
==============

The Danesfield project is organized as follows:

- `<danesfield>`_: directory where the Danesfield algorithmic modules
  live.
- `<tools>`_: directory with command line tools to execute
  the Danesfield algorithms.

Run Danesfield CLI
==================

The Danesfield pipeline can be run from a command line using
`tools/run_danesfield.py` and constructing a danesfield
configuration file based on the example in `<input.ini>`_.

If running via the docker container, first ensure you have the latest
danesfield docker image and nvidia-docker.  Use ``nvidia-docker``
to start a bash session inside the container:

.. code-block::

    nvidia-docker run -it --rm --gpus all --shm-size 8G\
     -v /$DATA:/mnt kitware/danesfield /bin/bash

where ``$DATA`` is a path on the host to data directory containing your input
imagery or point cloud and input.ini.  This host directory is mounted inside
the container at ``/mnt`` in the above command.

Once the environment is set up, you can execute a pipeline to process either
multiple satellite images or start with a geospatial point cloud.
To execute a pipeline with a point cloud, run

.. code-block::

   docker run --rm --gpus all -it -v ~/projects/danesfield:/root/danesfield -v ~/data/run_danesfield/:/root/run_danesfield kitware/danesfield 'LOGLEVEL=DEBUG python /root/danesfield//danesfield/tools/run_danesfield.py /root/run_danesfield/wrk/input.ini' > output_pointcloud.txt 2>&1


Note: `<input.ini>`_ should contain a valid point cloud path via ``p3d_fpath``.

To execute a pipeline with a set of satellite images, run

.. code-block::

    docker run --rm --gpus all -it -v ~/projects/danesfield:/root/danesfield -v ~/data/run_danesfield/:/root/run_danesfield kitware/danesfield 'LOGLEVEL=DEBUG python /root/danesfield/danesfield/tools/run_danesfield.py --image /root/run_danesfield/imageful/imageful.ini' >> output_image.txt 2>&1

Note: `<input.ini>`_ should contain a valid path to imagery via ``imagery_dir``.

See comments in `<input.ini>`_ for each configuration option.
To see more options on runnning danesfield pipeline, execute

.. code-block::

    python run_danesfield.py -h

where notable options are

- ``--image``: run pipeline with image data as the source; default uses a point cloud

- ``--roads``: get roads from open street maps; default extracts no roads

- ``--vissat``: run VisSat stereo pipeline using satellite imagery

- ``--run_metrics``: run evaluation metrics; requires ground truth for DSM, DTM, etc.

Minimum Harware Requirements
----------------------------

Danesfield runs a variety of processing steps, some of which take advantage of
multiple CPU cores and GPUs to accelerate processing large data sets.
At a minimum Danesfield requires:

- 12GB RAM, 16GB preferred
- a modern CPU preferably multi-core
- a modern Nvidia GPU with 8GB of GPU RAM, 16GB preferred

Project History
===============

The Danesfield project is named for Danesfield House in
Buckinghamshire, England.  This location was the center of `Allied
image intelligence <https://en.wikipedia.org/wiki/RAF_Medmenham>`_
during World War II.  During the war, analysts use multiple
overhead images to physically build 3D models of important sites.

Initial work on this project was funded by the
`IARPA CORE3D <https://www.iarpa.gov/index.php/research-programs/core3d>`_
program in 2017 and 2018.
The results of this initial work were presented at EarthVision_ 2019.
At the time of this publication, Danesfield included a dependency on
proprietary software developed by Raytheon.
The Raytheon P3D software was used to extract point clouds from satellite
images and bundle adjust RPC camera models.
This dependency limited the use Danesfield to users with a license for
the Raytheon P3D.

Since the initial CORE3D work, we have extended Danesfield in a few ways.
First, we have since integrated VisSat_ as an open source alternative to P3D
to allow for an end-to-end open source pipeline.
Note that while VisSat works as a replacement to P3D, several downstream
algorithms were trained on, or had parameters tuned for, P3D data.
So results produced with VisSat instead of P3D may not achieve the
same results as published.

Second, we have started to explore other input data in addition to WorldView 3.
We have generalized the pipeline to allow processing a geospatial point cloud
directly. This allows Danesfield to run on Lidar or other sources of point
clouds. We are also exploring other options, such as integrating the
`TeleSculptor <https://telesculptor.org/>`_ project to extract the point
cloud from aerial video sources.

Third, we are adding open source tools to convert the meshes produced by
Danesfield into the `3D Tiles <https://www.ogc.org/standards/3DTiles/>`_
format for more efficient transmission over the web.


.. _EarthVision: http://www.classic.grss-ieee.org/earthvision2019/
.. _EarthVisionPaper: http://openaccess.thecvf.com/content_CVPRW_2019/html/EarthVision/Leotta_Urban_Semantic_3D_Reconstruction_From_Multiview_Satellite_Imagery_CVPRW_2019_paper.html
__ EarthVisionPaper_
.. _VisSat: https://github.com/Kai-46/VisSatSatelliteStereo
