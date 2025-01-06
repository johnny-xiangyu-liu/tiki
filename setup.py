#  Copyright (c) [2025] [Xiangyu Liu]
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from setuptools import setup

setup(
    name='tiki',
    version='0.1',
    description='A Python module for AI training and evaluation using pytorch',
    author='Xiangyu Liu',
    author_email='johnny.xiangyu.liu@gmail.com',
    packages=['tiki'],
    install_requires=['torch'],
)
