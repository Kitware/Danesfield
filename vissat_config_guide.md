# VisSat Configuration Guide

```json
{
  "dataset_dir": "/path/to/image/data",
  "work_dir": "/path/to/work/directory",
  "bounding_box": {
    "zone_number": 17,
    "hemisphere": "N",
    "ul_easting": 435525,
    "ul_northing": 3355525,
    "width": 1402.0,
    "height": 1448.0
  },
  "steps_to_run": {
    "clean_data": true,
    "crop_image": true,
    "derive_approx": true,
    "choose_subset": true,
    "colmap_sfm_perspective": true,
    "inspect_sfm_perspective": false,
    "reparam_depth": true,
    "colmap_mvs": true,
    "aggregate_2p5d": true,
    "aggregate_3d": true
  },
  "alt_min": -30.0,
  "alt_max": 120.0
}
```

The configuration file for VisSatSatelliteStereo requires all of the information shown in the example above, which defines an area of interest in Jacksonville, Florida. 

First, it requires paths to the work directory and the directory that contains panchromatic satellite images of the area of interest. Next, the area of interest needs to be defined. This includes the 
UTM zone and hemisphere that the AOI is located within. It also includes the width and height of the AOI (in meters), and the coordinates (in UTM) of the upper left corner of the AOI. "Easting" 
corresponds to the area's longitude, and "Northing" to the latitude. The last piece of information needed by VisSat is the minimum and maximum altitudes in the area of interest. The altitudes do not 
need to be exact; in fact, it's better to provide a slightly greater altitude range to ensure that all no data is accidentally excluded.

To generate a point cloud to be used by the rest of the Danesfield pipeline, the steps to run should remain the same as in the example.

For more information about VisSatSatelliteStereo, see its [Github page](https://github.com/Kai-46/VisSatSatelliteStereo)
