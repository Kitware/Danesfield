# Danesfield Tools

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

## Third-party tools

### Core3D JSON data representation and parser

A data representation and meshing utility for CORE3D deliverables. See
https://github.com/CORE3D/data_rep_c3d.

Includes the following tools:

- `json2obj.py`
- `meshIO.py`
- `paramCounter.py`
- `primitiveMeshGen.py`
