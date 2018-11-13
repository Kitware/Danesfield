###############################################################################
# Copyright Kitware Inc. and Contributors
# Distributed under the Apache License, 2.0 (apache.org/licenses/LICENSE-2.0)
# See accompanying Copyright.txt and LICENSE files for details
###############################################################################


def get_hello_world():
    return "hello world"


def test_hello_word():
    assert get_hello_world() == "hello world"
