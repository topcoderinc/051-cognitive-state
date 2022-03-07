#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from glob import glob
from os.path import basename, splitext
from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="topcoder_cognitive_state",
    version="0.0.1",
    packages=find_packages(),
    py_modules=[
        splitext(basename(path))[0] for path in glob("topcoder_cognitive_state/*.py")
    ],
    install_requires=requirements,
)
