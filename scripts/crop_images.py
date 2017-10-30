from gaia.geo.processes_raster import *
from gaia.geo.geo_inputs import *

import os
import fnmatch

import traceback
import sys

src_root_dir = '/performer_source_data/wpafb/satellite_imagery'
dst_root_dir = '/performer_source_data/cropped/wpafb-D1/'

# pad the crop by the following percentage in width and height
padding_percentage = 10

### jacksonville
#ul_lon = -81.67078466333165
#ul_lat = 30.31698808384777

#ur_lon = -81.65616946309449
#ur_lat = 30.31729872444624

#lr_lon = -81.65620275072482
#lr_lat = 30.329923847788603

#ll_lon = -81.67062242425624
#ll_lat = 30.32997669492018

### ucsd
#ul_lon = -117.24298768132505
#ul_lat = 32.882791370856857

#ur_lon = -117.24296375496185
#ur_lat = 32.874021450913411

#lr_lon = -117.2323749640905
#lr_lat = 32.874041569804469

#ll_lon = -117.23239784772379
#ll_lat = 32.882811496466012

### wpafb D1
ul_lon = -84.11236693243779
ul_lat = 39.77747025512961

ur_lon = -84.10530109439955
ur_lat = 39.77749705975315

lr_lon = -84.10511182729961
lr_lat = 39.78290042788092

ll_lon = -84.11236485416471
ll_lat = 39.78287156225952

### wpafb D2
#ul_lon = -84.08847226672408
#ul_lat = 39.77650841377968

#ur_lon = -84.07992142333644
#ur_lat = 39.77652166058358

#lr_lon = -84.07959205694203
#lr_lat = 39.78413758747398

#ll_lon = -84.0882028871317
#ll_lat = 39.78430009793551



lon_pad = ((ur_lon - ul_lon)/padding_percentage)/2
ul_lon = ul_lon - lon_pad
ur_lon = ur_lon + lon_pad
ll_lon = ll_lon - lon_pad
lr_lon = lr_lon + lon_pad

lat_pad = ((ur_lat - ul_lat)/padding_percentage)/2
ul_lat = ul_lat - lat_pad
ur_lat = ur_lat + lat_pad
ll_lat = ll_lat - lat_pad
lr_lat = lr_lat + lat_pad

working_dst_dir = dst_root_dir

for root, dirs, files in os.walk(src_root_dir):
    for file_ in files:
        new_root = root.replace(src_root_dir, dst_root_dir)
        if not os.path.exists(new_root):
            os.makedirs(new_root)


        ext = os.path.splitext(file_)[-1].lower()
        if ext != ".ntf":
            continue

        try:
            src_img_file = os.path.join(root, file_)
            dst_img_file = os.path.join(new_root, file_)
            globaltemp = RasterFileIO(uri=src_img_file)
            polygonio1 = FeatureIO(features=
                                   [{"geometry":
                                         {"type":
                                              "Polygon", "coordinates":
                                              [[[ul_lon, ul_lat], [ur_lon, ur_lat],
                                                [lr_lon, lr_lat], [ll_lon, ll_lat],
                                                [ul_lon, ul_lat]]]},
                                     "properties":
                                         {"id":
                                              "North polygon"}} ])

            subset_pr = SubsetProcess(inputs=[globaltemp, polygonio1])

            # We called lower to cleanly get the ext so try to replace both
            subset_pr.output.uri = dst_img_file.replace(ext, '.tiff')
            subset_pr.output.uri = dst_img_file.replace(ext.upper(), '.tiff')
            print "dst_img_file: " + subset_pr.output.uri
            subset_pr.compute()
        except:
            print 'Problem cropping image: ' + src_img_file
            print traceback.format_exc()
