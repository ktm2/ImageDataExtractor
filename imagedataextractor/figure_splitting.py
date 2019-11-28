# -*- coding: utf-8 -*-
"""
Photo Detection
===============

Detect photos in figures.

@authors: Matt Swain and Ed Beard

"""

from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from .grid_splitting import *
from .photo_splitting import get_photos

import cv2
import glob
import io
import csv
from skimage import io as skio
from skimage.color import gray2rgb
from skimage import img_as_float
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import logging

log = logging.getLogger(__name__)

# New file paths
# photo_det_dir = os.path.dirname(os.path.abspath(__file__))
# input_dir = os.path.join(photo_det_dir, 'input')
# output_dir = os.path.join(photo_det_dir, 'output')
# csv_input_dir = os.path.join(photo_det_dir, 'csv_input')
# csv_output_dir = os.path.join(photo_det_dir, 'csv_output')
#
# img_dir_name = 'SEM%20rutile_040419'
# input_imgs = os.path.join(input_dir, img_dir_name)
# output_imgs = os.path.join(output_dir, img_dir_name)
# csv_input_path = os.path.join(csv_input_dir, img_dir_name + '_imgs.csv')
# csv_output_path = os.path.join(csv_output_dir, img_dir_name + '.csv')


def imread(f):
    """Read an image from a file.

    :param string|file f: Filename or file-like object.
    :return: Image array.
    :rtype: numpy.ndarray
    """

    # Ensure we use PIL so we can guarantee that imread will accept file-like object as well as filename
    img = skio.imread(f, plugin='pil')

    # Transform greyscale images to RGB
    if len(img.shape) == 2:
        log.debug('Converting greyscale image to RGB...')
        img = gray2rgb(img)

    # Transform all images pixel values to be floating point values between 0 and 1 (i.e. not ints 0-255)
    # Recommended in skimage-tutorials "Images are numpy arrays" because this what scikit-image uses internally
    img = img_as_float(img)

    return img


def imsave(f, img):
    """Save an image to file.

    :param string|file f: Filename or file-like object.
    :param numpy.ndarray img: Image to save. Of shape (M,N) or (M,N,3) or (M,N,4).
    """
    # Ensure we use PIL so we can guarantee that imsave will accept file-like object as well as filename
    skio.imsave(f, img, plugin='pil')


def split_by_photo(input_imgs, csv_input_path, output_imgs='', csv_output_path='', split=False):
    """ Identifies and segments photo areas
    :param bool multithreaded: Runs in parallel across all processors if True
    :param bool split: Leave true to save each segmented image separately, False to label and display bboxes on one image


    """
    log.info('Running plot area evaluation')

    log.info('Creating output directory if it doesnt exist...')
    if not os.path.exists(output_imgs):
        os.makedirs(output_imgs)

    log.info('Deleting output from previous run')
    files = [os.path.join(output_imgs, f) for f in os.listdir(output_imgs) if os.path.isfile(os.path.join(output_imgs, f))]
    for f in files:
        os.remove(f)

    log.info('Reading sample input from: %s' % csv_input_path)
    inf = io.open(csv_input_path, 'r')
    sample_csvreader = csv.reader(inf)
    log.info('Writing output to: %s' % csv_output_path)
    outf = open(csv_output_path, 'w')
    output_csvwriter = csv.writer(outf)

    # Load every image in the evaluation sample
    log.info('Loading images from: %s' % input_imgs)
    next(sample_csvreader)  # Skip header

    # Run the extraction using a multiprocessing pool for parallel execution (if needed)

    if split:
        results = [run_worker_split_images(row, input_imgs, output_imgs) for row in sample_csvreader]
    else:
        results = [run_worker(row, input_imgs, output_imgs) for row in sample_csvreader]

    # Write all results to output csv
    output_csvwriter.writerow(['fig_id', 'plot_id', 'left', 'right', 'top', 'bottom'])
    for im in results:
        for row in im:
            output_csvwriter.writerow(row)

    inf.close()
    outf.close()


def run_worker(row, input_imgs, output_imgs):
    """Detect photos in a figure, save an overlay file, and return output CSV rows."""
    img_format = '.' + row[2].split('.')[-1]
    filename = row[0].split('.')[0] + '_' + row[1] + img_format

    log.info('Processing: %s' % filename)

    # Load the image
    impath = os.path.join(input_imgs, filename)
    img = imread(impath)

    # Detect photos
    photos = get_photos(img)

    output_rows = []

    plt.rcParams['image.cmap'] = 'gray'
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)

    for i, photo in enumerate(photos):
        photo_id = i + 1

        if photo.area < 50000:
            log.warning("Pixel number : %s . Would be rejected in full pipeline." % photo.area )
            pass
        else:

            # Generate overlay for each detected plot area
            ax.add_patch(patches.Rectangle((photo.left, photo.top), photo.width, photo.height, alpha=0.2, facecolor='r'))
            ax.text(photo.left, photo.top + photo.height / 4, '[%s]' % photo_id, size=photo.height / 5, color='r')
            # Generate output CSV row
            output_rows.append([row[0], photo_id, photo.left, photo.right, photo.top, photo.bottom])

    dpi = fig.get_dpi()
    fig.set_size_inches(img.shape[1] / float(dpi), img.shape[0] / float(dpi))
    plt.savefig(os.path.join(output_imgs, '{}_{}.png'.format(row[0][:-5],row[1])))
    plt.close()

    return output_rows

def run_worker_split_images(row, input_imgs, output_imgs):
    """Detect photos in a figure, save all images to separate files"""

    img_format = '.' + row[2].split('.')[-1]
    filename = row[0].split('.')[0] + '_' + row[1] + img_format

    # Convert GIF images to PNG for output(cannot process gif in OpenCV)
    if img_format == '.gif':
        img_format = '.png'

    log.info('Processing: %s' % filename)

    # Load the image
    impath = os.path.join(input_imgs, filename)
    img = imread(impath)

    # Detect photos
    photos = get_photos(img)

    output_rows = []
    for i, photo in enumerate(photos):
        photo_id = i + 1
        if photo.area < 50000:
            log.warning("Pixel number : %s . Rejecting." % photo.area )
            pass
        else:

            log.info("Pixel number : %s" % photo.area)

            out_img = img[photo.top:photo.bottom,photo.left:photo.right]

            img_output_path = os.path.join(output_imgs, '{}_{}_{}'.format(row[0].split('.')[0], row[1], photo_id) + img_format)
            imsave(img_output_path, out_img)

            # Generate output CSV row
            output_rows.append([row[0], photo_id, photo.left, photo.right, photo.top, photo.bottom])

    return output_rows


def split_by_grid(input_imgs, output_imgs=''):
    """ Splits all input figures by detecting regular grid structrues """

    log.info('Creating output directory if it doesnt exist...')
    if not os.path.exists(output_imgs):
        os.makedirs(output_imgs)

    imgs = [os.path.join(input_imgs, f) for f in os.listdir(input_imgs) if os.path.isfile(os.path.join(input_imgs, f))]

    for img in imgs:
        split_fig_by_grid(img, output_imgs)


def split_fig_by_grid(figname, output_dir, eval_fig = False):
    '''Splits figures mined from publications into their constituent images. Note: Must be used on the 
    products of photo splitting.

    :param string figname: Name of the input figure.
    :param bool eval_fig: Optionally output an annotated version of the input for evaluation.

    :return list fig_split_final: list of constituent images as numpy.ndarrays.
    '''

    fig = cv2.imread(figname)

    #fig already grayscale.
    if len(fig.shape)==2:
        gfig=fig
    #fig is in color, convert to grayscale.
    else:
        gfig=cv2.cvtColor(fig,cv2.COLOR_BGR2GRAY)

    #Splitting process.
    fig_split_final, evaluation_fig = line_detection_and_split(gfig,eval_img = eval_fig)
    
    #Optional writing of images.
    if fig_split_final is not None and eval_fig == True:

        cv2.imwrite("eval"+ "_" +str(figname).split("/")[-1],evaluation_fig)

    index = 0
    for fig in fig_split_final:
        cv2.imwrite(os.path.join(output_dir, str(index) + "_" +str(figname).split("/")[-1]),fig)
        index += 1


    return fig_split_final



