#Inputs 2D image of particles, outputs radial distribution function of particles.
#Karim Mukaddem, Molecular Engineering Group, Department of Physics, University of Cambridge.

#NOTES:
# Some inconsistenies exist in [0],[1]/ x,y indexing of pixels, due to differences in how functions
# in different libraries behave.
# Changing all pixel acessing from a np based approach to cv2's built in (img.item.set) will greatly
# increase efficiency.

# TODO:
# Dark particles on light backgrounds?

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

    if len(img) * len(img[0]) < 50000:
        print("img smaller than 50k")
        return None,None

    scale, inlaycoords = scalebar_identification(img, testing = imgname)
    filteredvertices = particle_identification(img, inlaycoords, testing = imgname)

    #If less than 3 particles are found, redo analysis with inverted colors.
    if len(filteredvertices) < 3:
        filteredvertices_inverted = particle_identification(img, inlaycoords, testing = imgname, invert = True)

        if len(filteredvertices_inverted) > 0:
            if len(img.shape) == 2:
                gimg = img
            else:
                gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            rows = len(img)
            cols = len(img[0])

            if len(filteredvertices) > 0: 
                arealist = particle_metrics_from_vertices(img, gimg, rows, cols, filteredvertices)[1]
            else:
                arealist = []

            arealist_inv = particle_metrics_from_vertices(img, gimg, rows, cols, filteredvertices_inverted)[1]

            #If more overall area is attributed to particles in the inverted form, that version is passed to 
            #the calculation steps, both images get written out.
            if sum(arealist_inv) > sum(arealist):
                filteredvertices = filteredvertices_inverted


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

    resemblances, conclusion = match_to_shapes(filteredvertices)

    #Calculate particle metrics.
    colorlist, arealist, avgcolormean, avgcolorstdev, avgarea = particle_metrics_from_vertices(img, gimg, rows,
        cols, filteredvertices)
    #Convert from pixel square to meter square.
    avgarea = avgarea * (scale ** 2)
    #arealist = [a*(scale**2) for a in arealist]

    #particle_size_histogram(arealist,imgname)

    #number of particles.
    number_of_particles = len(filteredvertices)

    #Log for file.
    outfile = open(imgname.split('/')[-1].split(".")[0] + ".txt", "w")

    outfile.write(imgname.split('/')[-1] + "\n")
    #Need to add info about DOI, figure, material etc.
    outfile.write("Particle Number, Size in Pixels" + "\n")
    particle_index = 1
    for area in arealist:
        outfile.write(str(particle_index) + ", " + str(area) + "\n")
        particle_index+=1
    outfile.write("\n" + str(number_of_particles) + " particles detected." + "\n")
    outfile.write("Particle resemblances to regular shapes: " + "\n")
    outfile.write(str(resemblances) + "\n")
    outfile.write(conclusion + "\n")
    outfile.write("Average particle size: " + str(avgarea) + " sqm")

    outfile.close()

    #Calculate rdf.
    xRDF = []
    yRDF = []
    #if len(filteredvertices) > 9:
        #xRDF, yRDF = calculate_rdf(filteredvertices, rows, cols, scale, increment = 100, progress = True)
        #Output rdf.
        #output_rdf(xRDF, yRDF, imgname)

    return

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
            print("Scale and particle detection begun on: " + str(imgname))

            filteredvertices, scale = main_detection(imgname)

            if len(filteredvertices) > 0:
                after_detection(imgname, filteredvertices, scale)
            

    return


path_to_images = "/Users/karim/Desktop/evaluation_images/merged/2_karim_split/0_C3RA40414E_fig2_1.png"

#circles
#0_C3RA40414E_fig2_1
#0_C3RA40414E_fig2_2
#0_C6CE00824K_fig2_1


path_to_secondary = None

# path_to_images = "/home/batuhan/Documents/PhD Physics/Projects/imagedataextractor130219_2/2_karim_split/*.png"

# path_to_secondary = "/home/batuhan/Documents/PhD Physics/Projects/imagedataextractor130219_2/3_scalebar/false_negative/*.png"

run(path_to_images, path_to_secondary)






