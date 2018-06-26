#python train.py lcj_resnet34_1x1080_retrain.json /data/CORE3D/AOIS/DaytonJacksonville/pngdata
python -u train.py lcj_resnet34_1x1080_retrain.json /data/CORE3D/AOIS/Dayton_20sqkm/pngdata | tee trainlog.txt
#python train.py lcj_denseunet_1x1080_retrain.json /data/CORE3D/AOIS/DaytonJacksonville/pngdata
#python train.py lcj_denseunet_1x1080_retrain.json /data/CORE3D/AOIS/Dayton_20sqkm/pngdata
#python train.py lcj_extensionunet_1x1080_retrain.json /data/CORE3D/AOIS/DaytonJacksonville/pngdata
