##Columbia Building Segmentation Model for Core3D

It takes rgb dsm dtm and msi image as input and output a building mask. 

The model file is on [Kitware Data](https://data.kitware.com/#collection/5fa1b59350a41e3d192de2d5/folder/5fa1b65f50a41e3d192de73a)


Test:

```
python building_segmentation_test.py 
--rgb_image=/dvmm-filer2/projects/Core3D/D2_WPAFB/ortho-images/ortho_rgb_D2_07NOV16WV031100016NOV07165023.tif
--msi_image=/dvmm-filer2/projects/Core3D/D2_WPAFB/ortho-images/ortho_ps_D2_07NOV16WV031100016NOV07165023.tif
--dsm=/dvmm-filer2/projects/Core3D/D2_WPAFB/DSMs/D2_P3D_DSM.tif
--dtm=/dvmm-filer2/projects/Core3D/D2_WPAFB/DTMs/D2_DTM.tif
--model_path=../Inception_model/Dayton_best
--save_dir=../data/thres_img/
--output_tif
```

See options in building\_segmentation\_test.py for details.
