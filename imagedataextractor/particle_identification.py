#Author: Karim Mukaddem

from .correction_steps import *
from .scale_reading import *


def particle_identification(img, inlaycoords, testing = False, blocksize = 151, blursize = 3, invert = False):
    '''Runs contour detection and particle filtering
    functions on SEM images.
    
    :param numpy.ndarray img: input image.
    :param list inlaycoords: list of tuples, (x,y,w,h) top left corner, width and 
    height of inlays, including scalebar.
    :param int blocksize: parameter associated with image thresholding.
    :param int blursize: parameter associated with image thresholding.
    :param bool testing: Displays step by step progress for debugging.
    :param bool invert: Invert colors of image, useful for dark particles on light background.

    :return list filteredvertices: List of vertices of particles in image.
    :return list particlediscreteness: List of discreteness index for each particle.
    '''

    #Check if img already grayscale, if not convert.
    if len(img.shape) == 2:
        gimg = img
    else:
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Crop around edges of image.
    img = crop_image(img)
    gimg = crop_image(gimg)

    if invert == True:
        gimg = (255-gimg)

    #Calculate image metrics.
    rows, cols, imgarea, imgmean, imgstdev, crossstdev = image_metrics(gimg)

    #Initial contour detection.
    filteredvertices = find_draw_contours_main(img,gimg,blocksize,rows,cols,blursize, testing = testing)

    particlediscreteness = []

    if len(filteredvertices) > 0:
        #Calculate particle metrics.
        colorlist, arealist, avgcolormean, avgcolorstdev, avgarea = particle_metrics_from_vertices(img, gimg, rows,
        	cols, filteredvertices, invert)

        #Eliminate false positives.
        filteredvertices, arealist = false_positive_correction(filteredvertices, arealist, colorlist, avgcolormean,
        	avgcolorstdev, testing = testing, gimg = gimg)

        #Break up large clusters.
        filteredvertices = cluster_breakup_correction(filteredvertices, rows, cols, arealist, avgarea, blocksize, testing = testing, detailed_testing = False)

        #Eliminate particles that touch edges or inlays.
        filteredvertices, _ = edge_correction(filteredvertices, particlediscreteness = None, rows = rows, cols = cols, inlaycoords = inlaycoords, testing = testing, gimg = gimg)

        #Ellipse fitting.
        filteredvertices, particlediscreteness = discreteness_index_and_ellipse_fitting(filteredvertices, img, rows,
        	cols, imgstdev, testing = testing)

        #Eliminate particles that touch edges or inlays.
        filteredvertices, particlediscreteness = edge_correction(filteredvertices, particlediscreteness, rows, cols, inlaycoords, testing = testing, gimg = gimg)



    return filteredvertices, particlediscreteness