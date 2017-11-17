"""Setup script for Danesfield."""

from setuptools import setup, find_packages

from danesfield import __version__


with open('README.md') as f:
    desc = f.read()

setup(
    name='Danesfield',
    version=__version__,
    description='Algorithms for 3D reconstruction from satellite imagery',
    long_description=desc,
    author='Danesfield developers',
    author_email='kitware@kitware.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
    keywords='satellite, 3D, materials, geospatial',
    packages=find_packages(exclude=['tests*', 'docs']),
    package_data={
        'danesfield': [
            'conf/*',
        ]
    },
    url='https://gitlab.kitware.com/core3d/danesfield',
)
