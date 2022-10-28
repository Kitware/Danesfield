# Dockerfile for running an updated tiler script. We create a new
# conda environment for an updated VTK and a few other packages. We
# also update all danesfield scripts which run with the older
# libraries for most commands and with the new libraries for tiler.
#
# Rename original docker image (if not done already):
#   docker image tag kitware/danesfield kitware/danesfield-20220125
#
# Build:
#   docker build -t kitware/danesfield . -f tiler.dockerfile
#
# Run:
#   docker run \
#     -i -t --rm \
#     -v /path/to/data:/mnt/data \
#     kitware/danesfield \
#     <command>
# where <command> is like:
#   /danesfield/tools/generate-dsm.py ...
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
#     /danesfield/tools/material_classifier.py --cuda ...


FROM kitware/danesfield-20220125

LABEL maintainer="Kitware Inc. <kitware@kitware.com>"

# update NVIDIA keys
# see https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# Install additional packages
RUN apt-get update -q && \
    apt-get install -y -q \
        build-essential \
        libarchive-dev \
        vim \
        openssh-client

ARG CHANNELS="-c conda-forge/label/cf202003 -c defaults"
ARG CONDA=/opt/conda/bin/conda
RUN ${CONDA} install anaconda-client conda-build -y -q || exit 1

# Update conda and install mamba which works much, much faster than conda
RUN ${CONDA} update conda -y -q && \
    ${CONDA} install conda-libmamba-solver
ARG SOLVER="--experimental-solver libmamba"
RUN ${CONDA} install -n base conda-forge::mamba -y -q || exit 1
ARG MAMBA=/opt/conda/bin/mamba

RUN mkdir /root/conda-recipes
COPY conda-recipes/recipes/conda_build_config.yaml /root/conda-recipes/conda_build_config.yaml
WORKDIR /root/conda-recipes

ADD conda-recipes/recipes/vtk /root/conda-recipes/vtk
RUN ${MAMBA} create --name tiler --no-default-packages
RUN ${MAMBA} build ${CHANNELS} -m conda_build_config.yaml vtk
RUN ${CONDA} install  -n tiler -c local ${CHANNELS} vtk gdal pyproj

# Upgrade all danesfield scripts
RUN rm -rf /danesfield
COPY . /danesfield

RUN /opt/conda/bin/conda init bash
WORKDIR /

# Set entrypoint to script that sets up and activates CORE3D environment
ENTRYPOINT ["/bin/bash", "/danesfield/docker-entrypoint.sh"]

# # Set default command when executing the container
CMD ["/bin/bash"]

