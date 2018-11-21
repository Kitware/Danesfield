# Danesfield Tools

This directory contains several command line tools for executing the Danesfield algorithms.  The `run_danesfield.py` tool runs each component of the Danesfield system end-to-end.  Each of the following subsections covers a single tool.

Note that several of the tools require model files not included in this repository.  These model files may not be publicly available at the time of writing.  Please contact Kitware directory to request access to the models.

## Run Danesfield

This script runs each of the Danesfield system components in an integrated end-to-end pipeline.  The inputs / outputs of the script are controlled by a single configuration file.  A template configuration file (`input.ini`) can be found in this repositories root directory.

### Tools

- `run_danesfield.py`

### Prerequisites

A completed configuration file.  The configuration file requires paths to source materials, paths to model files, and parameters for some algorithms.

### Usage

```bash
python run_danesfield.py \
       <input_configuration_file>
```

## Segmentation by Height

### Tools

- `segment_by_height.py`

### Prerequisites

### Usage

```bash
python segment_by_height.py \
       <dsm_image_path> \
       <dtm_image_path> \
       <output_mask_path> \
       --msi <msi_image_path>
```

Can optionally pass in OpenStreetMap road data as a geojson file to include roads in the output mask.

```bash
python segment_by_height.py \
       <dsm_image_path> \
       <dtm_image_path> \
       <output_mask_path> \
       --msi <msi_image_path> \
       --road-vector <road_vector_geojson_path> \
       --road-rasterized <output_road_raster> \
       --road-rasterized-bridge <output_road_bridge_raster>
```

## Columbia Building Segmentation

### Tools

- `building_segmentation.py`

### Prerequisites

Download the files in [this folder](
https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5b3be0568d777f2e62259362).
When running the script, specify the path to this folder using the `--model_dir` argument
and the common prefix of the model files using the `--model_prefix` argument, i.e. "Dayton_best".

### Usage

```bash
python building_segmentation.py \
    --rgb_image <rgb_image_path> \
    --msi_image <msi_image_path> \
    --dsm <dsm_image_path> \
    --dtm <dtm_image_path> \
    --model_dir <model_directory> \
    --model_prefix <model_prefix> \
    --save_dir <output_directory> \
    --output_tif
```

## UNet Semantic Segmentation

### Tools

- `kwsemantic_segment.py`

### Prerequisites

Download the pretrained model
[here](https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5b4dfb0d8d777f2e6225b8da).
A default configuration file is included in this repository at
`danesfield/segmentation/semantic/test_denseunet_1x1080_retrain.json`.

### Usage

```bash
python kwsemantic_segment.py \
    <config_file_path> \
    <model_path> \
    <rgb_image_path> \
    <dsm_image_path> \
    <dtm_image_path> \
    <msi_image_path> \
    <output_directory> \
    <output_filename_prefix>
```

## Material Classification

Material classification from Rutgers University.

### Authors

- Matthew Purri (<matthew.purri@rutgers.edu>)

### Input

- Orthorectified image (GeoTIFF)
- Image metadata (.IMD or .tar)

### Output

- Orthorectified material segmentation map (GeoTIFF)
- Colorized material segmentation map (PNG)

### Tools

- `material_classifier.py`

### Prerequisites

Download [RN18_All.pth.tar](
https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5ab3b3a18d777f068578ecb0).
When running the script, specify the path to this file using the `--model-path` argument.

### Usage

```bash
python material_classifier.py --image_paths <image_paths> --info_paths <info_paths> --output_dir <output_dir> --model_path <model_path> --cuda
```

## PointNet Geon Extraction

PointNet Geon Extraction provided by Columbia University.

### Authors

- Xu Zhang (<xu.zhang@columbia.edu>)

### Input

- Building point cloud (las text)

### Output

- Building point cloud with root type labels (las text)

### Tools

- `roof_segmentation.py`

### Prerequisites

Download the files in [this folder](https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5b68a3fe8d777f06857c1f24).
When running the script, specify the path to this folder using the `--model_dir` argument
and the common prefix of the model files using the `--model_prefix` argument, i.e. "dayton_geon".

### Usage

```bash
python roof_segmentation.py \
    --model_dir=<path_to_model_dir> \
    --model_prefix=<model_prefix> \
    --input_pc=<path_to_input_pointcloud> \
    --output_txt=<path_to_output_pointcloud> \
    --output_png=<path_to_output_graphic> \
```

## Curve Fitting

Curve fitting provided by Columbia University.

### Authors

- Xu Zhang (<xu.zhang@columbia.edu>)

### Input

- Point cloud with roof type labels (las text)

### Output

- Point cloud of remaining (planar) points (las text)
- Geon file of fitted curves (npy)

### Tools

- `fitting_curved_plane.py`

### Prerequisites

### Usage

```bash
python fitting_curved_plane.py \
    --input_pc=<path_to_input_pointcloud> \
    --output_png=<path_to_output_png> \
    --output_txt=<path_to_output_remainingpoints_pointcloud> \
    --output_geon=<path_to_geon_output>
```

## Geon to mesh

Script to convert geon file to mesh provided by Columbia University.

### Authors

- Xu Zhang (<xu.zhang@columbia.edu>)

### Input

- Geon file (npy)
- DTM file (tif)

### Output

- Mesh file (ply)

### Tools

- `geon_to_mesh.py`

### Prerequisites

### Usage

```bash
python geon_to_mesh.py \
    --input_geon=<path_to_input_geon_npy> \
    --input_dtm=<path_to_input_dtm> \
    --output_mesh=<path_to_output_mesh_ply>
```

## Roof geon extraction

Wrapper script for running Purdue's point cloud segmentation and
reconstruction code, and Columbia's roof segmentation and geon fitting
code in the right sequence.

### Authors

- Bo Xu (<xu1128@purdue.edu>)
- Xu Zhang (<xu.zhang@columbia.edu>)

### Input

- P3D Point cloud (las)
- Threshold CLS file (tif)
- DTM file (tif)
- Roof segmentation model dir / prefix

### Output

- Mesh files (ply, obj)
- Geon JSON (json)

### Tools

- `roof_geon_extraction.py`

### Prerequisites

Download the files in [this folder](https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5b68a3fe8d777f06857c1f24).
When running the script, specify the path to this folder using the `--model_dir` argument
and the common prefix of the model files using the `--model_prefix` argument, i.e. "dayton_geon".

### Usage

```bash
python roof_geon_extraction.py \
    --las=<path_to_p3d_pointcloud> \
    --dtm=<path_to_input_dtm> \
    --cls=<path_to_input_threshold_cls> \
    --model_prefix=<prefix_for_model_files> \
    --model_dir=<directory_containing_model_files> \
    --output_dir=<path_to_output_directory>
```

## Get road vector

Fetches road vector data from OpenStreetMap for an AOI, and converts to GeoJSON.

### Input

- Latitude / Longitude bounds
- Output directory

### Output

- Road vector data (GeoJSON)
- Original OSM data (osm)

### Tools

- `get_road_vector.py`

### Usage

```bash
python get_road_vector.py \
       --left <left_bound> \
       --bottom <bottom_bound> \
       --right <right_bound> \
       --top <top_bound> \
       --output-dir <path_to_output_directory>
```

## Buildings to DSM

Renders a DSM or CLS from a DTM and polygons representing buildings.

### Input

- DTM file (tif)
- Building polygons (vtp or list of obj paths)

### Output

- DSM or CLS file (tif)

### Tools

- `buildings_to_dsm.py`

### Usage

```bash
python buildings_to_dsm.py \
       <path_to_dtm> \
       <path_to_output_file> \
       --input_obj_paths <list_of_obj_paths>
```

## Third-party tools

### Core3D JSON data representation and parser

A data representation and meshing utility for CORE3D deliverables. See
https://github.com/CORE3D/data_rep_c3d.

Includes the following tools:

- `json2obj.py`
- `meshIO.py`
- `paramCounter.py`
- `primitiveMeshGen.py`
