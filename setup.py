"""Setup script for Danesfield."""

from setuptools import setup, find_packages


def parse_version(package):
    """
    Statically parse the version number from __init__.py

    CommandLine:
        python -c "import setup; print(setup.parse_version('danesfield'))"
    """
    from os.path import dirname, join
    import ast
    init_fpath = join(dirname(__file__), package, '__init__.py')
    with open(init_fpath) as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if target.id == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_requirements(fname='requirements.txt'):
    """
    Parse the package dependencies listed in a requirements file.

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    from os.path import dirname, join, exists
    require_fpath = join(dirname(__file__), fname)
    if exists(require_fpath):
        with open(require_fpath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = [line for line in lines if not line.startswith('#')]
            return lines


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.md')
    if exists(readme_fpath):
        with open(readme_fpath, 'r') as f:
            return f.read()


if __name__ == '__main__':

    setup(
        name='Danesfield',
        version=parse_version('danesfield'),
        description='Algorithms for 3D reconstruction from satellite imagery',
        long_description=parse_description(),
        author='Danesfield developers',
        author_email='kitware@kitware.com',
        license='Apache 2.0',
        install_requires=parse_requirements('requirements.txt'),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
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
