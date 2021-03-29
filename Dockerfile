# Dockerfile for CORE3D Danesfield environment.
#
# Optional requirements:
#   nvidia-docker2 (https://github.com/NVIDIA/nvidia-docker)
#
# Build:
#   docker build -t core3d/danesfield .
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

LABEL maintainer="Max Smolens <max.smolens@kitware.com>"

COPY patches /patches

# Install prerequisites
RUN apt-get update && apt-get install -y software-properties-common && apt-get update && \
  add-apt-repository -y ppa:ubuntu-toolchain-r/test && add-apt-repository -y ppa:ubuntugis/ppa && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  make \
  git \
  bzip2 \
  ca-certificates \
  curl \
  libgl1-mesa-glx \
  libglu1-mesa \
  libxt6 && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/Kai-46/ColmapForVisSat.git && \
  cd ColmapForVisSat && \
  git checkout 696399e && \
  git apply ../patches/colmap_deps.patch

RUN git clone https://github.com/Kai-46/VisSatSatelliteStereo.git && \
  cd VisSatSatelliteStereo && \
  git checkout e5ca3a0 && \
  git apply ../patches/vissat.patch

ENV CONDA_EXECUTABLE /opt/conda/bin/conda

# Download and install miniconda3
# Based on https://github.com/ContinuumIO/docker-images/blob/fd4cd9b/miniconda3/Dockerfile
# The step to update 'conda' is necessary to avoid the following error when
# downloading packages (see https://github.com/conda/conda/issues/6811):
#
#     IsADirectoryError(21, 'Is a directory')
#

RUN curl --silent -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ${CONDA_EXECUTABLE} update -n base conda && \
    ${CONDA_EXECUTABLE} clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Copy environment definition first so that Conda environment isn't recreated
# unnecessarily when other source files change.
COPY ./deployment/conda/conda_env.yml \
     ./danesfield/deployment/conda/conda_env.yml

# Create CORE3D Conda environment
RUN ${CONDA_EXECUTABLE} env create -f ./danesfield/deployment/conda/conda_env.yml -n core3d && \
    ${CONDA_EXECUTABLE} clean -tipsy

# Install core3d-tf_ops package from kitware-danesfield / defaults
RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
  conda activate core3d && \
  pip install pip==20.0.1 && \
  conda install -c kitware-danesfield -c kitware-danesfield-df -y core3d-tf_ops && \
  conda clean -tipsy"]

# Install opencv package from conda-forge
RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
  conda activate core3d && \
  conda install -c conda-forge -y opencv && \
  conda clean -tipsy"]

RUN chmod +x pipeline.sh && \
  chmod +x /ColmapForVisSat/ubuntu1804_install_dependencies.sh && \
  chmod +x /ColmapForVisSat/ubuntu1804_install_colmap.sh && apt-get update && \
  /ColmapForVisSat/ubuntu1804_install_dependencies.sh && \ 
  cd /ColmapForVisSat && \
  ./ubuntu1804_install_colmap.sh

RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
  conda activate core3d && \
  pip install -r /VisSatSatelliteStereo/requirements.txt"]

RUN git clone https://github.com/LAStools/LAStools.git && \
  cd LAStools && \
  make

# Install Danesfield package into CORE3D Conda environment
COPY . ./danesfield
RUN rm -rf ./danesfield/deployment
RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
  conda activate core3d && \
  pip install -e ./danesfield"]

COPY ply2txt.py ply2txt.py

# Set entrypoint to script that sets up and activates CORE3D environment
ENTRYPOINT ["/bin/bash", "./danesfield/docker-entrypoint.sh"]

# Set default command when executing the container
CMD ["/bin/bash"]