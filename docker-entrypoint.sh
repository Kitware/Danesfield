#!/bin/bash
set -eo pipefail

# logging
echo "Execute docker-entrypoint.sh"

# Make 'conda' command available to shell environment
source /opt/conda/etc/profile.d/conda.sh

# Activate CORE3D environment
conda activate core3d

# Set up X virtual framebuffer (Xvfb) to support running VTK with OSMesa
# Note: requires mesalib variant of vtk package
Xvfb :1 -screen 0 1024x768x16 -nolisten tcp > xvfb.log &
export DISPLAY=:1.0

# Run specified command
eval "$@"
