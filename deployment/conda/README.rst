###################################
Setup Core3D development environent
###################################

This document describes how to setup a development environment for the
Core3D using Conda. Please follow the instructions step-by-step.
Currently, the environment is tested only on Mac OS and Linux and not on
Windows. It is possible that future version of environment may not work
on Windows because of the binary packages not available on Windows platform.

Install Conda
=============
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






