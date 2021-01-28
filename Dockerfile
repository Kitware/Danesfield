# Dockerfile for CORE3D Danesfield environment.
#
# Optional requirements:
#   nvidia-docker2 (https://github.com/NVIDIA/nvidia-docker)
#
# To run with CUDA support, ensure that nvidia-docker2 is installed on the host
#


FROM nvidia/cuda:9.0-devel-ubuntu16.04
LABEL maintainer="Max Smolens <max.smolens@kitware.com>"

# Install prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    curl \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxt6 \
    xvfb \
    sudo && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Download and install miniconda3
# Based on https://github.com/ContinuumIO/docker-images/blob/fd4cd9b/miniconda3/Dockerfile
# The step to update 'conda' is necessary to avoid the following error when
# downloading packages (see https://github.com/conda/conda/issues/6811):
#
#     IsADirectoryError(21, 'Is a directory')
#
ENV CONDA_EXECUTABLE /opt/conda/bin/conda
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
      conda install -c kitware-danesfield -c kitware-danesfield-df -y core3d-tf_ops && \
      conda clean -tipsy"]

# Install opencv package from conda-forge
RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
      conda activate core3d && \
      conda install -c conda-forge -y opencv && \
      conda clean -tipsy"]

# Install Danesfield package into CORE3D Conda environment
COPY . ./danesfield
RUN rm -rf ./danesfield/deployment

RUN apt-get update
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev

RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal

RUN ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && \
      conda activate core3d && \
      pip install --upgrade pip && \
      pip install -e ./danesfield"]

# Set entrypoint to script that sets up and activates CORE3D environment
ENTRYPOINT ["/bin/bash", "./danesfield/docker-entrypoint.sh"]

# Set default command when executing the container
CMD ["/bin/bash"]
