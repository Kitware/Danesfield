# Synopsis

• Deep learning solution for semantic segmentation, which is developed by Chengjiang Long at Kwiate. 
The current solutions are based on UNet architecture and here are the three variant versions:

(1). ExtensionUNet (extend the original UNet a little)

(2). ResenetUNet (use ResNet as encoder)

(3). DenseUNet (use the components of DenseNet, i.e., dense block and transition layer, to replace the 
covolution layers and maxpooling layers, respectively)

• The current solutions utilize the ensemble strategy with multiple folder of training data, the final 
prediction is merged by each model corresponding to each folder.



# Contributors

Chengjiang Long (chengjiang.long@kitware.com) -- algorithm development and implementation



## Inputs / Outputs

Input: RGB image, NDSM (DSM-DTM) and NDVI for each 2048x2048 tile.
Output: a probability map and a predition mask to indicate pixel-level label, i.e., building or
non-building. 



## Prerequisites

The requirements for are: 

PyTorch
opencv-python
sklearn
ubelt
tensorboardX
sympy
tqdm
numpy
scipy
gdal
pdal
rasterio

To install the requirements, you can run:

pip install -r requirements.txt



## Testing uage

``$ cd tools "
``$ bash denseunet_test.sh"

after running the above command line, you will get the prediction results in the temporal folder
``./tools/tmpresults/results/denseunet_/merged".
