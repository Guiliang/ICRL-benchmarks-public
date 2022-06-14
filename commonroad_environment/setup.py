#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""commonroad_rl setup file."""

import sys
from os import path

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Please install or upgrade setuptools or pip to continue")
    sys.exit(1)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

with open(path.join(this_directory, 'environment.yml'), encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name="commonroad-rl",
    version="2022.1",
    packages=find_packages(
        exclude=["tests", "utils_run"]
    ),
    package_data={"": ["*.xml", "*.pickle"]},
    description="Tools for applying reinforcement learning on commonroad scenarios.",
    long_description=readme,
    long_description_content_type="text/markdown",
    test_suite="commonroad_rl.tests",
    keywords="autonomous automated vehicles driving motion planning".split(),
    url="https://commonroad.in.tum.de/",
    install_requires=[],
    extras_require={
        "utils_run": ["optuna", "PyYAML"],
        "tests": ["pytest"],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Operating System :: POSIX :: Linux"
    ],
)
