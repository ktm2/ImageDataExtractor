# -*- coding: utf-8 -*-
"""
ImageDataExtractor
~~~~~~~~~~~~~~~~~
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

__title__ = 'ImageDataExtractor'
__version__ = '0.0.1'
__author__ = 'Karim Mukaddem'

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from .extract import extract_document, extract_documents, extract_image, extract_images