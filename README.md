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


#### Install ChemDataExtractor-IDE

In order to use ImageDataExtractor first install the bespoke version of ChemDataExtractor, [ChemDataExtractor-IDE](https://github.com/edbeard/chemdataextractor-ide). 

Clone the repository by running:

    $ git clone https://github.com/edbeard/chemdataextractor-ide.git

and install with:

    $ python setup.py install
    
Then download the required machine learning models with:

    $ cde data download

*See https://github.com/edbeard/chemdataextractor-ide for more details* 

#### Install ImageDataExtractor

__NOTE : Upon release IDE will be installed using pip (by simply running `pip install imagedataextractor`). While in development, follow the instructions below instead.__

Now to install ImageDataExtractor, clone the repository with:

    $ git clone  https://github.com/ktm2/ImageDataExtractor.git
    
Then create a wheel file by running:

    $ python setup.py bdist_wheel
    
*You may have to run `pip install wheel` if this fails.*
    
Then install using pip:

    $ pip install dist/ImageDataExtractor-0.0.1-py3-none-any.whl  
    
  
 
## Running the code

__Full documentation on running the code can be found at www.imagedataextractor.com .__

Open a python terminal and run 

    >>> import imagedataextractor as ide
    
Then run:

    >>> ide.extract_document(<path/to/document>)
    
to automatically identify and extract the images from a document. Full details on supported input and output formats can be found at www.imagedataextractor.com . 