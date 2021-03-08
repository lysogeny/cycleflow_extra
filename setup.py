#!/usr/bin/env python3
"""Setup file for cfextra"""

from setuptools import setup, find_packages

with open("README.md") as f:
    README = f.read()

#with open("LICENSE") as f:
#    LICENSE = f.read()

setup(
    name="cycleflow_extra",
    version="0.0.1",
    description="Extra tools for cycleflow",
    long_description=README,
    author="Jooa Hooli",
    author_email="code@jooa.xyz",
    url="https://github.com/lysogeny/cfextra",
#    license=LICENSE,
    packages=find_packages(exclude=("tests", "docs")),
)

