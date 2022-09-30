import os
import sys
import subprocess
import boto3
from collections import defaultdict
import argparse
import configparser
import json
import shutil

def run(args):
	parser = argparse.ArgumentParser(description="Download WV3 images to run \
		the Danesfield pipeline")
	parser.add_argument('--site', help='CORE3D site to download images for. \
		This will be used as a prefix for naming output directories and files,\
		 even if Danesfield is starting with a point cloud.',
		default='Jacksonville')
	parser.add_argument('--point_cloud', help='Start Danesfield with the point \
		cloud at this path instead of generating one from images')
	parser.add_argument('--img_path', help='Location where WV3 images should be \
		downloaded to', default=f'{os.getcwd()}/imgpath')
	parser.add_argument('--run', help='Run the Danesfield pipeline', 
						action='store_true')
	parser.add_argument('--out_path', help='Location that Danesfield should use \
		as an output directory', default='outdir')
	parser.add_argument('--config_dir', help='Location where Danesfield \
		configuration files can be found. You must use input_<region>.ini and \
		<region>_config.json for Danesfield and Vissat config names, respectively.',
		 default='configdir')
	parser.add_argument('--show_sites', help='List available CORE3D sites and exit', 
						action='store_true')
	parser.add_argument('--model_path', help='Location of necessary model files \
		for Danesfield', default="Models")
	parser.add_argument('--key_id', help='AWS key ID')
	parser.add_argument('--secret_key', help='AWS secret key')
	parser.add_argument('--pull_image', help='Pull Danesfield image from \
		Kitware\'s repository', action='store_true')
	parser.add_argument('--pull_models', help='Pull models from data.kitware.com',
		action='store_true')
	args = parser.parse_args(args)

	dset = 'spacenet-dataset'
	bucket = 'Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/'
	client = boto3.client('s3', aws_access_key_id=args.key_id, 
		aws_secret_access_key=args.secret_key)

	# Pull Danesfield image if requested
	if args.pull_image:
		print('Pulling docker image')
		cmd_args = ['docker', 'pull', 'kitware/danesfield']
		subprocess.run(cmd_args)
		print('Completed pulling docker image')

	# Pull Danesfield's models if requested
	if args.pull_models and not os.path.exists(args.model_path):
		print('Pulling Models File')
		os.makedirs(args.model_path, exist_ok=True)
		cmd_args = ['wget', '-O', f'./models.zip', 
			'https://data.kitware.com/api/v1/folder/5fa1b5e150a41e3d192de52b/download']
		subprocess.run(cmd_args)
		extract_command = ['unzip', f'models.zip']
		subprocess.run(extract_command)
		print('Models Pulled and Extracted')

	# Create necessary directories
	os.makedirs(args.out_path, exist_ok=True)
	os.makedirs(args.config_dir, exist_ok=True)

	if args.point_cloud==None:	
		os.makedirs(args.img_path, exist_ok=True)
		# determine which regions have imagery
		regions = []
		for p in client.list_objects(Bucket=dset, Prefix=bucket)['Contents']:
			regions.append(p['Key'].split('/')[3])
		regions = list(set(regions))
		wv3 = defaultdict(list)
		for r in regions:
			r_bucket = bucket + r + '/'
			for p in client.list_objects(Bucket=dset, Prefix=r_bucket)['Contents']:
				tmp = p['Key'].split('/')
				if tmp[4]=='WV3':
					wv3[r].append(p['Key'])

		# display valid regions and exit
		if args.show_sites:
			print("Sites with available WV3 imagery:", ', '.join(wv3.keys()))
			return 0

		# User selected invalid region
		if args.site not in wv3:
			print("Invalid site selected, please choose from amongst the following:",
				', '.join(wv3.keys()))
			return 1

		# download files for requested region
		for f in wv3[args.site]:
			os.makedirs(os.path.join(args.img_path, '/'.join(f.split('/')[3:-1])), 
				exist_ok=True)
			download_path = os.path.join(args.img_path, '/'.join(f.split('/')[3:]))
			if not os.path.exists(download_path):
				print(f'Downloading: {f}')
				client.download_file(dset, f, os.path.join(args.img_path,
					'/'.join(f.split('/')[3:])))
				print(f'Download complete: {f}')

	if args.run:
		print('Running Danesfield')

		img_path = os.path.abspath(args.img_path)
		config_dir = os.path.abspath(args.config_dir)
		out_path = os.path.abspath(args.out_path)
		model_path = os.path.abspath(args.model_path)
		cmd_args = ['docker', 'run', '-it', '--gpus', 'all', '--shm-size', '8G', \
					'-v', out_path+':/workdir', '-v', config_dir+':/configs', \
					'-v', model_path+':/models']

		# gsd and cuda will need to be filled out by user
		config = configparser.ConfigParser()
		config.read(os.path.join(args.config_dir, 'input_'+args.site+'.ini'))
		config['paths']['p3d_fpath'] = os.path.join('/workdir', args.site,
			'cloud.las')
		config['paths']['work_dir'] = os.path.join('/workdir', args.site)
		config['aoi']['name'] = args.site.lower()
		config['roof']['model_dir'] = "models/Columbia Geon Segmentation Model"
		config['roof']['model_prefix'] = 'dayton_geon'
		config['material']['model_fpath'] = \
			"/models/Rutgers Material Segmentation Model/model_D4_20.pth.tar"

		if args.point_cloud==None:		
			config['paths']['imagery_dir'] = os.path.join('/mnt', args.site, 'WV3')
			config['paths']['aoi_config'] = os.path.join('/configs', args.site+\
				'_config.json')
			with open(os.path.join(args.config_dir, args.site+'_config.json'),'r') as f:
				vissat_config = json.load(f)
			vissat_config['dataset_dir'] = os.path.join('/mnt', args.site, 'WV3', 'PAN')
			vissat_config['work_dir'] = os.path.join('/workdir', args.site, 'vissat')
			with open(os.path.join(args.config_dir, args.site+'_config.json'), 'w') as f:
				json.dump(vissat_config, f, indent=4)

			cmd_args.extend(['-v', img_path+':/mnt','kitware/danesfield', 
				'source /opt/conda/etc/profile.d/conda.sh && \
				conda activate core3d && python danesfield/tools/run_danesfield.py \
				--image --vissat --roads /configs/input_'+args.site+'.ini'])
		else:
			shutil.copyfile(args.point_cloud, os.path.join(args.out_path, 
				args.site,os.path.basename(args.point_cloud)))
			cmd_args.extend(['kitware/danesfield', 'source /opt/conda/etc/profile.d/conda.sh \
				&& conda activate core3d && python danesfield/tools/run_danesfield.py \
				--roads /configs/input_'+args.site+'.ini'])
			config['paths']['p3d_fpath'] = os.path.join('/workdir', args.site,
				os.path.basename(args.point_cloud))

		with open(os.path.join(args.config_dir, 'input_'+args.site+'.ini'), 'w') as f:
			config.write(f)

		print(' '.join(cmd_args))
		subprocess.run(cmd_args)

	return 0


if __name__=='__main__':
	exit(run(sys.argv[1:]))

# To avoid manually entering AWS credentials as arguments, users can install aws-cli
# See https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html for help

