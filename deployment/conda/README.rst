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

   mkdir CORE3D_DIR (pick a name of your choice)
   cd CORE3D_DIR
   git clone https://gitlab.kitware.com/core3d/danesfield.git
   cd danesfield
   conda env create -f deployment/conda/conda_env.yml
   source activate core3d-dev
   pip install -e .


Install Gaia
============

Gaia is a Python library aiming for geospatial analytics reusable components

.. code-block:: bash

   cd .. (assuming you are at the root level inside of CORE3D_DIR)
   git clone https://github.com/OpenDataAnalytics/gaia.git
   cd gaia
   pip install -r requirements-dev.txt
   pip install -e .


###################################
Test Core3D development environent
###################################

Invoke pytest and flake8 at the root level of the repository

.. code-block:: bash

   pytest (should pass all tests under tests sub directory)
   flake8 . (should pass all style checks)









