#from rdf_functions import *

from contour_detections import *
from correction_steps import *
from img_utils import *
from scale_reading import *


def particle_identification(img, inlaycoords, testing = None, blocksize = 151, blursize = 3):
    '''Runs contour detection and particle filtering
    functions on SEM images.
    
    :param numpy.ndarray img: input image.
    :param list inlaycoords: list of tuples, (x,y,w,h) top left corner, width and 
    height of inlays, including scalebar.
    :param int blocksize: parameter associated with image thresholding.
    :param int blursize: parameter associated with image thresholding.
    :param bool testing: Displays step by step progress for debugging.

    :return list filteredvertices: List of vertices of particles in image.
    '''

    #Check if img already grayscale, if not convert.
    if len(img.shape) == 2:
        gimg = img
    else:
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    #Crop around edges of image.
    img = crop_image(img)
    gimg = crop_image(gimg)

    #Calculate image metrics.
    rows, cols, imgarea, imgmean, imgstdev, crossstdev = image_metrics(gimg)

    #Initial contour detection.
    filteredvertices = find_draw_contours_main(img,blocksize,rows,cols,blursize, testing = False)

    if len(filteredvertices) == 0:
        return []

    #Calculate particle metrics.
    colorlist, arealist, avgcolormean, avgcolorstdev, avgarea = particle_metrics_from_vertices(img, gimg, rows,
    	cols, filteredvertices)

    #Eliminate false positives.
    filteredvertices, arealist = false_positive_correction(filteredvertices, arealist, colorlist, avgcolormean,
    	avgcolorstdev, testing = False, gimg = gimg)

    #Break up large clusters.
    filteredvertices = cluster_breakup_correction(filteredvertices, rows, cols, arealist, avgarea, blocksize, testing = False, detailed_testing = False)

    #Eliminate particles that touch edges or inlays.
    filteredvertices = edge_correction(filteredvertices, rows, cols, inlaycoords, testing = False, gimg = gimg)

    #Ellipse fitting.
    particlediscreteness, filteredvertices = discreteness_index_and_ellipse_fitting(filteredvertices, img, rows,
    	cols, imgstdev)

    #Eliminate particles that touch edges or inlays.
    filteredvertices = edge_correction(filteredvertices, rows, cols, inlaycoords, testing = False, gimg = gimg)

    if testing != None:
        for i in range(len(filteredvertices)):
            cv2.polylines(img,[filteredvertices[i]],True,(0,255,0),thickness=1)

        #show_image(img)
        
        cv2.imwrite("det_"+str(testing).split("/")[-1],img)

    return filteredvertices