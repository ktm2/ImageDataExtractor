"""
Identify TEM images from scientific articles using chemdataextractor

@author : Ed Beard

"""

from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from chemdataextractor import Document
import chemdataextractor
import urllib
import multiprocessing as mp

import os
import csv
import io
import sys
import logging

logging.basicConfig()
log = logging.getLogger(__name__)


class TEMImageExtractor():

    def __init__(self, input, output='', typ='tem'):
        self.input = input
        self.output = output
        self.img_csv_path = str(os.path.join(self.output, os.path.basename(self.output) + '_raw.csv'))
        self.docs = []
        self.paths = []
        self.imgs = []
        self.urls = []
        self.img_type = typ

        print(chemdataextractor.__file__)

    def get_img_paths(self):
        """ Get paths to all images """
        docs = os.listdir(self.input)
        self.docs = [(doc, os.path.join(self.input, doc)) for doc in docs]

        # Create image folders if it doesn't exist
        if not os.path.exists(os.path.join(self.output, 'raw_images')):
            os.makedirs(os.path.join(self.output, 'raw_images'))

    def get_img(self, doc):
        """Get images from doc using chemdataextractor"""

        # Load document image data from file
        tem_images = []
        cde_doc = Document.from_file(open(doc[1], "rb"))
        print('This article is : %s' % doc[0])
        imgs = cde_doc.figures
        del cde_doc

        # Identify relevant images from records
        for img in imgs:
            detected = False  # Used to avoid processing images twice
            records = img.records
            caption = img.caption
            for record in records:
                if detected is True:
                    break

                rec = record.serialize()
                if [self.img_type] in rec.values():
                    detected = True
                    print('%s instance found!' % self.img_type)
                    tem_images.append((doc[0], img.id, img.url.encode('utf-8'), caption.text.replace('\n', ' ').encode('utf-8')))

        if len(tem_images) != 0:
            return tem_images
        else:
            return None

    def download_image(self, url, file, id):
        """ Download all TEM images"""

        imgs_dir = os.path.join(self.output, 'raw_images')

        if len(os.listdir(imgs_dir)) <= 999999999:
            img_format = url[-3:]
            filename = file[:-5] + '_' + id + '.' + img_format
            path = os.path.join(imgs_dir, filename)

            print("Downloading %s..." % filename)
            if not os.path.exists(path):
                urllib.urlretrieve(url, path) # Saves downloaded image to file
            else:
                print("File exists! Going to next image")
        else:
            sys.exit()

    def save_img_data_to_file(self):
        """ Saves list of tem images"""

        imgf = open(self.img_csv_path, 'wb')
        output_csvwriter = csv.writer(imgf)
        output_csvwriter.writerow(['article', 'fig id', 'url', 'caption'])

        for row in self.imgs:

            # Ignore results without a URL
            if row[2] != '':
                output_csvwriter.writerow(row)

    def get_all_tem_imgs(self, parallel=True):
        """ Get all TEM images """

        self.get_img_paths()

        # Check if TEM images info found
        if os.path.isfile(self.img_csv_path):
            with io.open(self.img_csv_path, 'rb') as imgf:
                img_csvreader = csv.reader(imgf)
                next(img_csvreader)
                self.imgs = list(img_csvreader)
        else:

            # If not found, identify TEM images
            if parallel:
                pool = mp.Pool(processes=mp.cpu_count())
                tem_images = pool.map(self.get_img, self.docs)
            else:
                tem_images =[]
                for doc in self.docs:
                    try:
                        imgs = self.get_img(doc)
                        if imgs is not None:
                            tem_images.append(imgs)
                    except Exception as e:
                        print(e)

            self.imgs = [img for doc in tem_images if doc is not None for img in doc]

            self.save_img_data_to_file()
            print('%s image info saved to file' % self.img_type)

        # Download TEM images
        for file, id, url, caption in self.imgs:
            self.download_image(url, file, id)

    def get_tem_imgs(self):
        """ Get the TEM images for a single Document"""

        if not os.path.isfile(self.input):
            raise Exception('Input should be a single document for this method')

        # Create image folders if it doesn't exist
        if not os.path.exists(os.path.join(self.output, 'raw_images')):
            os.makedirs(os.path.join(self.output, 'raw_images'))

        # Check if TEM images info found
        if os.path.isfile(self.img_csv_path):
            with io.open(self.img_csv_path, 'rb') as imgf:
                img_csvreader = csv.reader(imgf)
                next(img_csvreader)
                self.imgs = list(img_csvreader)
        else:
            try:
                doc = (self.input.split('/')[-1], self.input)
                self.imgs = self.get_img(doc)
            except Exception as e:
                print(e)

        self.save_img_data_to_file()
        print('%s image info saved to file' % self.img_type)

        # Download TEM images
        for file, id, url, caption in self.imgs:
            self.download_image(url, file, id)