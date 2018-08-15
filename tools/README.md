# Danesfield Tools

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
