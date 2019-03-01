#from rdf_functions import *

from contour_detections import *
from correction_steps import *
from img_utils import *
from scale_reading import *


def particle_identification(img, inlaycoords, testing = None, blocksize = 151, blursize = 3, invert = False):
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
    filteredvertices = find_draw_contours_main(img,gimg,blocksize,rows,cols,blursize, testing = False)

    if len(filteredvertices) > 0:
        #Calculate particle metrics.
        colorlist, arealist, avgcolormean, avgcolorstdev, avgarea = particle_metrics_from_vertices(img, gimg, rows,
        	cols, filteredvertices, invert)

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
        drawing_img = img.copy()
        for i in range(len(filteredvertices)):
            cv2.polylines(drawing_img,[filteredvertices[i]],True,(0,255,0),thickness=1)
            #Annotate particle #.
            annotate = True
            if annotate == True:
                (xcom,ycom),contradius = cv2.minEnclosingCircle(filteredvertices[i])
                xcom=int(xcom)
                ycom=int(ycom)
                contradius=int(contradius)
                cv2.circle(drawing_img,(xcom,ycom),1,(0,0,255),1)
                cv2.putText(drawing_img,str(i+1),(xcom+3,ycom+3),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,255),thickness=1)           


        #show_image(img)

        if invert == True:
            cv2.imwrite("inv_det_"+str(testing).split("/")[-1],drawing_img)
        else:
            cv2.imwrite("det_"+str(testing).split("/")[-1],drawing_img)

    return filteredvertices