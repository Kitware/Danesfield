# Dockerfile for CORE3D Danesfield environment.
#
# Optional requirements:
#   nvidia-docker2 (https://github.com/NVIDIA/nvidia-docker)
#
# Build:
#   docker build -t kitware/danesfield .
#
# Run:
#   docker run \
#     -i -t --rm \
#     -v /path/to/data:/mnt/data \
#     core3d/danesfield \
#     <command>
# where <command> is like:
#   danesfield/tools/generate-dsm.py ...
#
# To run with CUDA support, ensure that nvidia-docker2 is installed on the host,
# then add the following argument to the command line:
#
#   --runtime=nvidia
#
# Example:
#   docker run \
#     -i -t --rm \
#     --runtime=nvidia \
#     core3d/danesfield \
#     danesfield/tools/material_classifier.py --cuda ...

FROM nvidia/cuda:10.0-devel-ubuntu18.04

LABEL maintainer="Kitware Inc. <kitware@kitware.com>"

# Install prerequisites
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
  rm -f /etc/apt/sources.list.d/nvidia-ml.list && \apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
  add-apt-repository -y ppa:ubuntugis/ppa && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  make \
  git \
  bzip2 \
  cmake \
  ca-certificates \
  curl \
  libgl1-mesa-glx \
  libglu1-mesa \
  libxt6 \
  nodejs \
  npm \
  xvfb \
  unzip \
  wget && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*

# update NVIDIA keys
# see https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# Build packages
RUN apt-get update -q && \
    apt-get install -y -q \
        build-essential \
        vim \
        openssh-client \
        libgl1-mesa-dev \
        libxt-dev

# Generate ssh key to access private repo
RUN ssh-keygen -q -t ed25519 -C 'danlipsa@danesfield-conda-build' -N '' -f /root/.ssh/id_ed25519 && \
    echo "Copy the following key to gitlab, Preferences, SSH Keys:" && \
    cat /root/.ssh/id_ed25519.pub && \
    ssh-keyscan -t rsa gitlab.kitware.com >> ~/.ssh/known_hosts && \
    sleep 30

# Download and install miniconda3
# Based on https://github.com/ContinuumIO/docker-images/blob/fd4cd9b/miniconda3/Dockerfile
# The step to update 'conda' is necessary to avoid the following error when
# downloading packages (see https://github.com/conda/conda/issues/6811):
#
#     IsADirectoryError(21, 'Is a directory')
#
ARG CONDA=/opt/conda/bin/conda
RUN curl --silent -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    ${CONDA} clean -tipsy && \
    rm ~/miniconda.sh

RUN ${CONDA} install anaconda-client conda-build -y -q || exit 1
# mamba works much, much faster than conda, set as default solver
RUN ${CONDA} update conda -y -q && \
    ${CONDA} install conda-libmamba-solver && \
    ${CONDA} config --set experimental_solver libmamba

RUN apt-get install -y -q libarchive-dev

WORKDIR /
# Patches for Colmap and VisSat
ADD patches /patches

# Install ColmapForVisSat from Github
RUN git clone --recursive https://github.com/Kai-46/ColmapForVisSat.git && \
  cd ColmapForVisSat && \
  git checkout 9d96671 && \
  git apply /patches/colmap_deps.patch && \
  chmod +x /ColmapForVisSat/ubuntu1804_install_dependencies.sh && \
  chmod +x /ColmapForVisSat/ubuntu1804_install_colmap.sh && \
  apt-get update && \
  /ColmapForVisSat/ubuntu1804_install_dependencies.sh && \
  cd /ColmapForVisSat && \
  ./ubuntu1804_install_colmap.sh

# Install VisSat package from Github
RUN ["/bin/bash", "-c", "git clone https://github.com/Kai-46/VisSatSatelliteStereo.git && \
  cd VisSatSatelliteStereo && \
  git checkout e5ca3a0 && \
  git apply ../patches/vissat.patch && \
  source /opt/conda/etc/profile.d/conda.sh && \
  conda create -n vissat python=3.6 pip=20.0.* && \
  conda activate vissat && \
  pip install -r /VisSatSatelliteStereo/requirements.txt && \
  pip uninstall -y numpy && \
  conda install -y numpy libgdal gdal"]

# Install LAStools package from Github
RUN git clone https://github.com/LAStools/LAStools.git && \
  cd LAStools && \
  make

RUN wget https://www.ipol.im/pub/art/2017/179/BilateralFilter.zip && \
  unzip /BilateralFilter.zip && \
  rm /BilateralFilter.zip && \
  cd BilateralFilter && \
  mkdir build && \
  cd build && \
  cmake -DCMAKE_BUILD_TYPE=Release .. && \
  make

# Install latest stable version of node and npm
RUN ["/bin/bash", "-c", "/usr/bin/npm cache clean -f && \
     /usr/bin/npm install -g n  && \
     n stable"]

# Install 3d-tiles-tools for converting glb to b3dm
RUN ["/bin/bash", "-c", "git clone https://github.com/CesiumGS/3d-tiles-validator.git && \
     cd 3d-tiles-validator/tools && \
     /usr/local/bin/npm install"]

# Install meshoptimizer for optimizing the gltf and converting to glb
RUN ["/bin/bash", "-c", "git clone https://github.com/zeux/meshoptimizer.git src && \
     mkdir meshoptimizer && \
     cd meshoptimizer && \
     mv ../src . && \
     cd src && \
     git checkout v0.16 && \
     cd .. && \
     mkdir build && \
     cd build && \
     cmake -DMESHOPT_BUILD_GLTFPACK:BOOL=ON ../src && \
     make -j"]

COPY deployment/conda/conda_env.yml /root/conda_env.yml

# Create CORE3D Conda environments
RUN ${CONDA} list && ${CONDA} env create -f /root/conda_env.yml -n core3d && \
    ${CONDA} create --name tiler --no-default-packages && \
    ${CONDA} create --name texture --no-default-packages && \
    ${CONDA} create --name roofgeon --no-default-packages && \
    ${CONDA} create --name roads -c conda-forge -c defaults openssl=1.1 pyopenssl=19.1 gdal=2.4 python=3.6 pyproj requests

RUN ln /usr/lib/x86_64-linux-gnu /usr/lib64 -s

ARG CHANNELS="-c conda-forge/label/cf201901 -c defaults"

# Add the conda recipes for the build
ADD conda-recipes/recipes /root/conda-recipes

WORKDIR /root/conda-recipes

RUN chmod 777 /root/conda-recipes/build_all.sh && \
    /root/conda-recipes/build_all.sh

RUN ${CONDA} install -n core3d -c local ${CHANNELS} liblas
RUN ${CONDA} install -n roofgeon -c local ${CHANNELS} python-pcl
RUN ${CONDA} install -n roofgeon -c kitware-danesfield ${CHANNELS} vtk=v9.1 python=3.6
RUN ${CONDA} install -n roofgeon -c local ${CHANNELS} core3d-purdue
RUN ${CONDA} install -n roofgeon -c local ${CHANNELS} core3d-tf_ops gdal tqdm shapely matplotlib tensorflow-gpu=1.13 cudatoolkit=10

RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
  conda activate roofgeon && \
  pip install plyfile"]

RUN ${CONDA} install -n core3d -c local ${CHANNELS} laspy
RUN ${CONDA} install -n core3d -c local ${CHANNELS} pubgeo-tools
RUN ${CONDA} install -n core3d -c local ${CHANNELS} pubgeo-core3d-metrics
RUN ${CONDA} install -n core3d -c kitware-danesfield ${CHANNELS} vtk=v9.1
RUN ${CONDA} install -n texture -c kitware-danesfield ${CHANNELS} vtk=v9.1
RUN ${CONDA} install -n texture -c local ${CHANNELS} kwiver vtk=v9.1 texture-atlas gdal scipy pyproj imageio python-pdal

ARG CHANNELS="-c conda-forge/label/cf202003 -c defaults"
RUN ${CONDA} install -n tiler -c local ${CHANNELS} vtk gdal pyproj

WORKDIR /

# Install Danesfield package into CORE3D Conda environment
COPY . ./danesfield
RUN rm -rf ./danesfield/deployment
RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
  conda activate roads && \
  pip install -e ./danesfield && \
  conda activate core3d && \
  pip install -e ./danesfield"]

# Clean up.
RUN ${CONDA} clean -tipy

# Set entrypoint to script that sets up and activates CORE3D environment
ENTRYPOINT ["/bin/bash", "./danesfield/docker-entrypoint.sh"]

# Set default command when executing the container
CMD ["/bin/bash"]
