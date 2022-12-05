export CHANNELS="-c conda-forge/label/cf201901 -c defaults"
/opt/conda/bin/conda build $CHANNELS -m conda_build_config.yaml liblas
/opt/conda/bin/conda build -c local $CHANNELS -m conda_build_config.yaml pcl
/opt/conda/bin/conda build -c local $CHANNELS -m conda_build_config.yaml python-pcl
/opt/conda/bin/conda build -c local $CHANNELS -c kitware-danesfield -m conda_build_config.yaml core3d-purdue
/opt/conda/bin/conda build $CHANNELS -m conda_build_config.yaml core3d-tf_ops
/opt/conda/bin/conda build $CHANNELS -m conda_build_config.yaml laspy
/opt/conda/bin/conda build -c local $CHANNELS -m conda_build_config.yaml pubgeo-tools
/opt/conda/bin/conda build -c local $CHANNELS -m conda_build_config.yaml pubgeo-core3d-metrics
/opt/conda/bin/conda build $CHANNELS -c kitware-danesfield -m conda_build_config.yaml texture-atlas
/opt/conda/bin/conda build $CHANNELS -m conda_build_config.yaml kwiver
export CHANNELS="-c conda-forge/label/cf202003 -c defaults"
/opt/conda/bin/conda build $CHANNELS -m conda_build_config.yaml vtk
