####################################
Setup Core3D Development Environment
####################################

This document describes how to setup a development environment for the
CORE3D Danesfield project using Conda. Please follow the instructions
step-by-step.

Currently, the environment is tested only on Mac OS and Linux and not on
Windows. The environment may not work on Windows because some binary packages
are not available on the Windows platform. For Windows 10 users, the Windows
Subsystem for Linux (WSL) allows you to run Linux within Windows.
This environment has been verified on Ubuntu 16.04 running within WSL.

Install Conda
=============
Conda3 is required to setup the environment using Python 3.  Follow the URL
below to install miniconda3 for your platform.  Please ensure that you install
Conda3 (for Python3) and not Conda2 or Anaconda.

https://conda.io/miniconda.html


Create Core3D Conda Environment
===============================

Due to some conflicting package dependencies between conda channels,
two packages must be installed manually after setting up the initial
environment.

.. code-block:: bash

   mkdir CORE3D_DIR (pick a name of your choice)
   cd CORE3D_DIR
   git clone https://gitlab.kitware.com/core3d/danesfield.git
   cd danesfield
   conda env create -f deployment/conda/conda_env.yml
   source activate core3d-dev
   conda install -c kitware-danesfield core3d-tf_ops
   conda install -c conda-forge opencv
   pip install -e .

To deactivate the core3d-dev environment, run:

.. code-block:: bash

   source deactivate

To remove the core3d-dev environment, run:

.. code-block:: bash

   source deactivate (if core3d-dev is activated before)
   conda remove --name core3d-dev --all

To update the core3d-dev environment when new packages have been added, run:

.. code-block:: bash

   source activate core3d-dev
   conda env update -f deployment/conda/conda_env.yml

###################################
Test Core3D Development Environment
###################################

Invoke pytest and flake8 at the root level of the repository

.. code-block:: bash

   pytest (should pass all tests under tests sub directory)
   flake8 . (should pass all style checks)
