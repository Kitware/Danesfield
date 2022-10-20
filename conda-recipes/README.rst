Danesfield Conda Recipes
============================

This repository contains conda recipes for libraries used in
danesfield that don't have conda packages available at conda-forge or
anaconda repositories.

Reuse kitware/danesfield docker and update only VTK
---------------------------------------------------
* Rename kitware/danesfield to kitware/danesfield-oldvtk and build
  a new container using ../danesfield/Dockerfile-updatevtk
  See the instructions in the aforementioned Dockerfile.


Build and upload conda packages
----------------------------------
* Start the docker container using instructions in file:
  danesfield-docker-conda-build/README.rst

* Inside the container:
* generate a SSH key into the container and add it to your gitlab account
  (make sure the container stays private - otherwise you give access to your gitlab
  account) so that you get access to gitlab private repos needed by some of the
  packages that we build

  >>> ssh-keygen -t ed25519 -C "danlipsa@danesfield-conda-build"
  >>> cat ~/.ssh/id_ed25519.pub

  Copy that to the clipboard and paste it to your gitlab account:
  Preferences > SSH Keys. This allows you to download protected repos.


* Build the recipes and upload them to kitware-danesfield conda channel::

    cd ~/danesfield/danesfield-conda-recipes/recipes
    sh build_all.sh
    cd ~/miniconda3/conda-bld/linux-64/
    ~/miniconda3/bin/anaconda upload -u kitware-danesfield *.tar.bz2
    cd ../noarch
    ~/miniconda3/bin/anaconda upload -u kitware-danesfield *.tar.bz2
    ~/miniconda3/bin/anaconda upload -u kitware-danesfield repodata.json

Copy core3d-dev conda packages to private channels
------------------------------------------------------

Use the instructions in danesfield/deployment/conda/README.rst to
create a core3d-dev environment.  Activate the environment and copy
all packages in that environment to a local folder::

    source activate core3d-dev
    ./copy-conda-pkgs.sh ~/public_html/conda

Three folders are going to be created which corresponds to the three
conda repositories we use: defaults, conda-forge and pytorch.  We have
two options:

1. We can upload all packages in these folders to three different
   organizations on anaconda. In this case we have replace defaults,
   conda-forge and pytorch in deployment/conda/conda_env.yml with the
   new organizations.

2. The packages saved this way can be used through a regular web
   server. In that case we have to execute::

     conda index ~/public_html/conda/defaults
     conda index ~/public_html/conda/conda-forge
     conda index ~/public_html/conda/pytorch

The three organizations in conda_env.yml have to be replaced with the
URLs for the folders that contain packages from the three
channels. For instance pytorch is replaced by
http://constanta/~danlipsa/conda/pytorch

Packages
--------

C++
core3d-purdue
core3d-tf_ops
flann
liblas
pcl
python-pcl
pubgeo-tools
texture-atlas
vtk

Python
gaia
laspy
pubgeo-core3d-metrics
