"""
MAT-Tools: Python Framework for Multiple Aspect Trajectory Data Mining

The present application offers a tool, to support the user in the clustering of multiple aspect trajectory data.It integrates into a unique framework for multiple aspects trajectories and in general for multidimensional sequence data mining methods.
Copyright (C) 2022, MIT license (this portion of code is subject to licensing from source project distribution)

Created on Apr, 2024
Copyright (C) 2024, License GPL Version 3 or superior (see LICENSE file)

@author: Yuri Santos, Tarlis Portela
"""
import setuptools

import configparser
config = configparser.ConfigParser()
config.read('pyproject.toml')
VERSION = config['project']['version']
PACKAGE_NAME = config['project']['name']
DEV_VERSION = "0.1b0"

VERSION = VERSION.replace('"', '')

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
#    version=DEV_VERSION,
    author="Yuri Nassar",
    author_email="yuri.nassar@gmail.com",
    description="Clustering Methods for Multiple Aspect Trajectory Data Mining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mat-analysis/mat-clustering",
#    packages=setuptools.find_packages(include=[PACKAGE_NAME, PACKAGE_NAME+'.*']),
    packages=setuptools.find_packages(),
#    include_package_data=True,
#    scripts=[
#         'scripts/x.py', # For future
#    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords='data mining, python, trajectory classification, trajectory analysis, clustering',
    license='GPL Version 3 or superior (see LICENSE file)',
)
