nvidia-docker run -it --rm --gpus all --shm-size 8G\
 -v /data/core3D-data:/mnt\
 -v $HOME:/home/$USER\
 -v $HOME/work/danesfield:/work\
 kitware/danesfield\
 /bin/bash
