# Tiler - Convert large 3D geospatial datasets to the 3D Tiles format.

# Install
- Install nodejs, npm and proj-bin packages on Ubuntu 20.04
- `cd ~/external/3d-tiles-tools/;npm install 3d-tiles-tools`. Help at: <https://github.com/AnalyticalGraphicsInc/3d-tiles-tools/tree/master/tools>
- `cd ~/external/gltf-pipeline;npm install gltf-pipeline`. Help at: <https://github.com/CesiumGS/gltf-pipeline>
- Clone <https://github.com/CesiumGS/3d-tiles-samples>. and then `npm install.` (only if you want to see datasets in Cesium)

# Convert data to 3D Tiles
- Create a gltf file for all Jacksonville OBJs, convert gltf to glb and glb to b3dm
```
./tools/tiler-test.sh -c jacksonville
```

# View in cesium
1. Use 3d-tiles-samples
  - Link the tileset created for previous set:
  `cd ~/external/3d-tiles-samples/tilesets; ln -s ~/projects/danesfield/danesfield/jacksonville-3d-tiles`
  - Start web server:
  `cd ..;npm start`
2. Load `cd ~/projects/VTK/src/IO/Cesium3DTiles;google-chrome jacksonville-3dtiles.html` created like in the documentation.


# TODO
- add saving to glb to vtkGLTFExporter
- add a b3dm writer to VTK
- add MTL support to the OBJ reader
