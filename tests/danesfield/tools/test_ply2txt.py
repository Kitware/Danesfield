###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################

from tools.ply2txt import convert
from os.path import dirname, join
from os import remove

data_dir = join(dirname(dirname(__file__)), 'data')


def test_ply2txt():
    ply_source = join(data_dir, 'octahedron.ply')
    txt_target = join(data_dir, 'octahedron.txt')
    convert([ply_source, txt_target])

    with open(txt_target, 'r') as f:
        lines = f.readlines()
        assert lines == ['-1.0 1.0 0.0\n',
                         '-1.0 -1.0 0.0\n',
                         '1.0 -1.0 0.0\n',
                         '1.0 1.0 0.0\n',
                         '0.0 0.0 0.7\n',
                         '0.0 0.0 -0.7\n']

    # clean up temporary output
    remove(txt_target)
