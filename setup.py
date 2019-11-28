#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = '''A toolkit for extracting particle data from microscopy images. '''

setup(
    name='ImageDataExtractor',
    version='1.0.1',
    author='Karim Mukaddem',
    author_email='ktm25@cam.ac.uk',
    license='MIT',
    url='https://github.com/ktm2/ImageDataExtractor',
    download_url='https://github.com/ktm2/ImageDataExtractor/archive/1.0.0.tar.gz',
    packages=find_packages(),
    description='A toolkit for extracting particle data from microscopy images.',
    long_description=long_description,
    keywords='image-mining mining chemistry cheminformatics microscopy SEM TEM html xml science scientific',
    zip_safe=False,
    include_package_data=True,
    setup_requires=['numpy'],
    install_requires=[
        'opencv-python', 'pillow', 'pytesseract', 'matplotlib==2.2.4', 'keras', 'tensorflow', 'scikit-image<0.15', 'marshmallow', 'numpy', 'cython', 'ChemDataExtractor-IDE'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)
