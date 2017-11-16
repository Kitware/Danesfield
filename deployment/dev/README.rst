###################################
Setup Core3D development environent
###################################

This document describes how to setup a development environment for the
Core3D using Conda. Please follow the instructions below step-by-step.

Install Conda
=============
https://conda.io/miniconda.html

Create Core3D Conda Environment
===============================

.. code-block:: bash

   conda env create -f conda_env.yml python=3.6.3
   source activate core3d-dev

Install Kitware Stack
=====================

1. Install Girder
-----------------

.. code-block:: bash

   git clone https://github.com/girder/girder.git
   cd girder
   pip install -e .
   girder-install web
   girder-server

Visit localhost:8080 and you should see Girder default home page
Exit out (ctrl-c) to install the point cloud plugin

2. Install Point Cloud Plugin
-----------------------------

.. code-block:: bash

   cd ..
   git clone https://github.com/OpenGeoscience/pointcloud_viewer.git
   girder-install plugin -s pointcloud_viewer
   girder-install web --dev --plugins pointcloud_viewer

Install the plugin after you create a Girder username as admin in the
Girder Admin console.

3. Install Gaia
-----------------------------

.. code-block:: bash
   cd ..
   git clone https://github.com/OpenDataAnalytics/gaia.git
   cd gaia
   pip install -r requirements-dev.txt
   pip install -e .











