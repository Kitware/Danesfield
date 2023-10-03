###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################
from danesfield.metrics.config import _current_path, get_filename, get_template, populate_template
from os.path import dirname, join

data_dir = join(dirname(dirname(__file__)), 'data')


def test__current_path():
    danesfield_root = dirname(dirname(dirname(data_dir)))
    assert _current_path() == join(danesfield_root, 'danesfield', 'metrics')


def test_get_filename():
    file1 = 'test_dsm.tif'
    file2 = 'test_cls.tif'
    assert get_filename(file1, file2) == 'test_dsm__test_cls.config'


def test_get_template():
    with open(join(data_dir, 'metrics-template-config.txt'), 'r') as f:
        contents = f.read()
        assert get_template() == contents


def test_populate_template():
    template = '$ref_prefix $test_dsm $test_cls $test_mtl $test_dtm'
    populated = populate_template(template, 'prefix', 'dsm', 'cls', 'mtl', 'dtm')
    assert populated == 'prefix dsm cls mtl dtm'
