#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2022 Bence Becsy
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Setup script for QuickCW
call should look like python setup.py install
"""
import os
from distutils.core import setup

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths
setup(
    name='QuickCW',
    version='0.0dev',
    requires=['numba', 'enterprise_extensions', 'numba', 'h5py'],
    packages=['QuickCW'],
    scripts=[],
    license='GPL',
    long_description=open('README.md').read(),
    )


