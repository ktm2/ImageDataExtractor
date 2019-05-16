# ImageDataExtractor

ImageDataExtractor is a toolkit for the automatic extraction of microscopy images. 

## Features

- Automatic detection and download of microscopy images from scientific articles 
- HTML and XML document format support
- High-throughput capabilities
- Direct extraction from image files 
- PNG, GIF, JPEG, TIFF image format support

## Installation

*NB: It is advised that all installations of ImageDataExtractor are run inside a virtual environment. Click [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for more information.* 

#### Install ImageDataExtractor
To install ImageDataExtractor, clone the repository with:

    git clone  https://github.com/ktm2/ImageDataExtractor.git
    
Then install the main dependencies by running:
    
    pip install -r requirements.txt
    
#### Install ChemDataExtractor-IDE

Next the user should install the bespoke version of ChemDataExtractor, [ChemDataExtractor-IDE](https://github.com/edbeard/chemdataextractor-ide). 

Clone the repository by running:

    git clone https://github.com/edbeard/chemdataextractor-ide.git

and install with:

    python setup.py install
    
Then download the required machine learning models with:

    cde data download

*See https://github.com/edbeard/chemdataextractor-ide for more details* 
 
## Running the code

