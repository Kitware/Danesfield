# Danesfield Tools

## Material Classification

Material classification from Rutgers University.

### Authors

- Matthew Purri (<matthew.purri@rutgers.edu>)

### Input

- Orthorectified image (GeoTIFF)
- Image metadata (.IMD)

### Output

- Orthorectified material segmentation map (GeoTIFF)

### Tools

- `material_classifier.py`

### Prerequisites

Download [RN18_All.pth.tar](
https://data.kitware.com/#collection/59c1963d8d777f7d33e9d4eb/folder/5ab3b3a18d777f068578ecb0)
 and place it in `danesfield/materials/pixel_prediction/architecture/`.

### Usage

```bash
python material_classifier.py <image_path> <imd_path> <output_path>
```
