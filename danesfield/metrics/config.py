#!/usr/bin/env python

###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


"""
Configuration file utilities for CORE3D metrics.
"""

import os
import re

from string import Template


def _current_path():
    """Return path of this file."""
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def get_filename(test_dsm, test_cls):
    """
    Get config filename based on test filenames.
    """
    def get_basename(filename):
        basename = os.path.basename(filename)
        return os.path.splitext(basename)[0]

    return '__'.join([
        get_basename(test_dsm),
        get_basename(test_cls)
    ]) + '.config'


def get_template():
    """
    Get metrics config file template contents.
    """
    template_file = os.path.join(_current_path(), 'metrics-template.config')
    with open(template_file, 'r') as f:
        contents = f.read()
    return contents


def populate_template(contents, ref_prefix, test_dsm, test_cls, test_mtl, test_dtm):
    """
    Populate metrics config template.
    """
    template = Template(contents)

    template = template.substitute({
        'ref_prefix': ref_prefix,
        'test_dsm': os.path.basename(test_dsm),
        'test_cls': os.path.basename(test_cls),
        'test_mtl': os.path.basename(test_mtl),
        'test_dtm': os.path.basename(test_dtm)
    })

    # remove unset parameters (lines ending with '= ')
    return re.sub('.*=[ \t]*\n', '', template)
