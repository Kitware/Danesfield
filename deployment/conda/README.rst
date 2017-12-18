###################################
Setup Core3D development environent
###################################

This document describes how to setup a development environment for the
Core3D using Conda. Please follow the instructions step-by-step.

Currently, the environment is tested only on Mac OS and Linux and not on
Windows. The environment may not work on Windows because some binary packages
are not available on the Windows platform. For Windows 10 users, the Windows
Subsystem for Linux (WSL) allows you to run Linux within Windows.
This environment has been verified on Ubuntu 16.04 running within WSL.

Install Conda
=============
Conda3 is required to setup the environment using Python 3.  Follow the URL
below to install miniconda3 for your platform.  Plase ensure that you install
Conda3 (for Python3) and not Conda2 or Anaconda.

https://conda.io/miniconda.html


Create Core3D Conda Environment
===============================

.. code-block:: bash

   mkdir CORE3D_DIR (pick a name of your choice)
   cd CORE3D_DIR
   git clone https://gitlab.kitware.com/core3d/danesfield.git
   cd danesfield
   conda env create -f deployment/conda/conda_env.yml
   source activate core3d-dev
   pip install -e .

To deactivate the core3d-dev environent, run:

.. code-block:: bash

   source deactivate

To remove the core3d-dev environent, run:

.. code-block:: bash

   source deactivate (if core3d-dev is activated before)
   conda remove --name core3d-dev --all


###################################
Test Core3D development environent
###################################

Invoke pytest and flake8 at the root level of the repository

.. code-block:: bash

   pytest (should pass all tests under tests sub directory)
   flake8 . (should pass all style checks)

#####################
Some Useful Resources
#####################

GDAL/OGR cookbook: https://pcjericks.github.io/py-gdalogr-cookbook/

Workshop: Raster and vector processing with GDAL: http://download.osgeo.org/gdal/workshop/foss4ge2015/workshop_gdal.pdf






