# setup.py
"""
This script sets up the package for installation using setuptools.

Dependencies:
- torch
- torchvision
- PyYAML
- numpy

Usage:
    python setup.py install
"""
from setuptools import setup, find_packages

setup(
    name='multimodal_transformer',
    version='0.1',
    packages=find_packages(),
    install_requires=['torch','torchvision','PyYAML','numpy'],
)