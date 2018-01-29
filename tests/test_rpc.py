from danesfield.rpc import RPCModel, rpc_from_gdal_dict

# sample RPC metadata read by GDAL from the following 2016 MVS Benchmark image
# 01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.NTF
rpc_md = {
  'HEIGHT_OFF': '31',
  'HEIGHT_SCALE': '501',
  'LAT_OFF': '-34.4732',
  'LAT_SCALE': '0.0708',
  'LINE_DEN_COEFF': '1 0.0001912806 0.0005166397 -1.45044e-05 '
                    '-3.860133e-05 2.634582e-06 -4.551145e-06 6.859296e-05 '
                    '-0.0002410782 9.753265e-05 -1.456261e-07 5.310624e-08 '
                    '-1.913253e-05 3.18203e-08 3.870586e-07 -0.000206842 '
                    '9.128349e-08 0 -2.506197e-06 0 ',
  'LINE_NUM_COEFF': '0.0002703625 0.04284488 1.046869 0.004713542 '
                    '-0.0001706129 -1.525177e-07 1.255623e-05 -0.0005820134 '
                    '-0.000710512 -2.510676e-07 3.179984e-06 3.120413e-06 '
                    '3.19923e-05 4.194369e-06 7.475295e-05 0.0003630791 '
                    '0.0001021649 4.493725e-07 3.156566e-06 4.596505e-07 ',
  'LINE_OFF': '21477',
  'LINE_SCALE': '21478',
  'LONG_OFF': '-58.6096',
  'LONG_SCALE': '0.09279999999999999',
  'MAX_LAT': '-34.4378',
  'MAX_LONG': '-58.5632',
  'MIN_LAT': '-34.5086',
  'MIN_LONG': '-58.656',
  'SAMP_DEN_COEFF': '1 0.0003374458 0.0008965622 -0.0003730697 '
                    '-2.666499e-05 -2.711356e-06 5.454434e-07 4.485658e-07 '
                    '2.534922e-05 -4.546709e-06 0 -1.056044e-07 '
                    '-5.626866e-07 2.243313e-08 -2.108053e-07 9.199534e-07 '
                    '0 -3.887594e-08 -1.437016e-08 0 ',
  'SAMP_NUM_COEFF': '0.006585953 -1.032582 0.001740937 0.03034485 '
                    '0.0008819178 -0.000167943 0.0001519299 -0.00626254 '
                    '-0.00107337 9.099077e-06 2.608985e-06 -2.947004e-05 '
                    '2.231277e-05 4.587831e-06 4.16379e-06 0.0003464555 '
                    '3.598323e-08 -2.859541e-06 5.159311e-06 -1.349187e-07 ',
  'SAMP_OFF': '21249',
  'SAMP_SCALE': '21250'
}

# sample geographic points (lon, lat, alt) that correspond to the image above
points = [[-58.589407278263572, -34.492834551467631, 20.928231142319902],
          [-58.589140738420539, -34.492818509990848, 21.9573811423199],
          [-58.588819506933184, -34.492808611762605, 27.1871011423199],
          [-58.58855693683482, -34.492802905977392, 19.2657311423199],
          [-58.58839238727699, -34.49280925602671, 26.606641142319901]]


def test_rpc_init():
    RPCModel()


def test_rpc_from_gdal_dict():
    rpc_from_gdal_dict(rpc_md)


def test_rpc_projection():
    model = rpc_from_gdal_dict(rpc_md)
    print(model.project(points[0]))


def test_rpc_multi_projection():
    model = rpc_from_gdal_dict(rpc_md)
    print(model.project(points))


def test_rpc_back_projection():
    model = rpc_from_gdal_dict(rpc_md)
    img_pt = model.project(points[0])
    bp = model.back_project(img_pt, points[0][2])
    print("diff: ", bp - points[0])


def test_rpc_multi_back_projection():
    model = rpc_from_gdal_dict(rpc_md)
    img_pts = model.project(points)
    print("Running multi-point back projection")
    bp = model.back_project(img_pts, [p[2] for p in points])
    print("diff: ", bp - points)
