Danesfield Conda Recipes
============================

This repository contains conda recipes for libraries used in
danesfield that don't have conda packages available at conda-forge or
anaconda repositories.

Reuse kitware/danesfield docker and update only dependecies for tiler
---------------------------------------------------
* tiler is the 3D Tiles conversion script. See instructions in ../tiler.dockerfile
  for building the docker image. We rebuild VTK and install a new conda environment
  called tiler. run_danesfield.py tries to run tiler in a new conda
  environment called tiler.


Rebuild kitware/danesfield
--------------------------
* See instructions in ../Dockerfile. This is not currently able to
resolve all conda packges. An issue that needs to be resolved is
updating python-pcl which is currently not maintained.
* run_danesfiled.py needs to be updated to execute tiler in the same conda environment.
