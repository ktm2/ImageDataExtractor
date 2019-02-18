#Inputs 2D image of particles, outputs radial distribution function of particles.
#Karim Mukaddem, Molecular Engineering Group, Department of Physics, University of Cambridge.

#NOTES:
# Some inconsistenies exist in [0],[1]/ x,y indexing of pixels, due to differences in how functions
# in different libraries behave.
# Changing all pixel acessing from a np based approach to cv2's built in (img.item.set) will greatly
# increase efficiency.

# TODO:
# Edge correction and hierarchical relations for fitted ellipses.
# Inlay determination.

from scalebar_identification import *
from particle_identification import *
from rdf_functions import *

import glob


def main_detection(imgname):
    '''Where the detection happens.

    :param sting imgname: name of image file.

    :return list filteredvertices: list of vertices of particles in image.
    :return float scale: Scale of pixels in image (m/pixel).
    '''

    img = cv2.imread(imgname)

    scale, inlaycoords = scalebar_identification(img, testing = imgname)

    filteredvertices = particle_identification(img, inlaycoords, testing = imgname)

    return filteredvertices, scale

def after_detection(imgname, filteredvertices, scale):
    '''After detection has happened calculate particle metrics and RDF.

    :param sting imgname: name of image file.
    :param list filteredvertices: list of vertices of particles (as numpy.ndarray) in image.
    :param float scale: Scale of pixels in image (m/pixel).

    :return float avgarea: average size of particles (m2)
    :return float avgcolormean: average pixel intensity in particles.
    :return list rdf: [x,y] columns of particle RDF.

    '''

    img = cv2.imread(imgname)

    rows = len(img)
    cols = len(img[0])

    #Check if img already grayscale, if not convert.
    if len(img.shape) == 2:
        gimg = img
    else:
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Calculate particle metrics.
    colorlist, arealist, avgcolormean, avgcolorstdev, avgarea = particle_metrics_from_vertices(img, gimg, rows,
        cols, filteredvertices)

    #Convert from pixel square to meter square.
    avgarea = avgarea * (scale ** 2) 

    #Calculate rdf.
    xRDF, yRDF = calculate_rdf(filteredvertices, rows, cols, scale, increment = 100, progress = True)

    return avgarea, avgcolormean, [xRDF,yRDF]

def run(path_to_images, path_to_secondary = None):
    '''Runs scalebar and particle identification.

    :param string path_to_images: path to images of interest.
    :param string path_to_secondary: path to secondary directory, useful
    to skip certain images (if starting or restarting a batch) or process only
    a preapproved set. 

    '''
    images = []
    images.extend(glob.glob(path_to_images))

    secondary = []
    if path_to_secondary != None:
        #Index value 9 here depends on what precedes your filename.
        #9 is to ignore "scalebar_" prefix.
        secondary.extend([a.split('/')[-1][9:] for a in glob.glob(path_to_secondary)])

    for imgname in images:
        if imgname.split('/')[-1] not in secondary:
            print "Scale and particle detection begun on: " + str(imgname)

            filteredvertices, scale = main_detection(imgname)

            avgarea, avgcolormean, [xRDF, yRDF] = after_detection(imgname, filteredvertices, scale)

            #Output rdf.
            plot_rdf(xRDF, yRDF, imgname)    

    return


path_to_images = "./input_images/*.png"
path_to_secondary = None

run(path_to_images, path_to_secondary)






