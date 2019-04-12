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

    filteredvertices, particlediscreteness = particle_identification(img, inlaycoords, testing = False)

    inverted = False

    #This is disabled until further development.
    #If less than 3 particles are found, redo analysis with inverted colors.
    if False:
        rows = len(img)
        cols = len(img[0])

        if len(img.shape) == 2:
            gimg = img
        else:
            gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        if len(filteredvertices) > 0:
            arealist = particle_metrics_from_vertices(img, gimg, rows, cols, filteredvertices)[1]
            detection_ratio = sum(arealist)/float(rows*cols)
            mean_particlediscreteness = sum(particlediscreteness)/float(len(particlediscreteness))
        else:
            detection_ratio = 0
            mean_particlediscreteness = 0
            arealist = []

        if len(filteredvertices) < 3 or detection_ratio < 0.1 or mean_particlediscreteness < 30:
            filteredvertices_inverted, particlediscreteness_inv = particle_identification(img, inlaycoords, testing = True, invert = True)

            if len(filteredvertices_inverted) > 0:
                if len(img.shape) == 2:
                    gimg = img
                else:
                    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                rows = len(img)
                cols = len(img[0])

                arealist_inv = particle_metrics_from_vertices(img, gimg, rows, cols, filteredvertices_inverted)[1]
                mean_particlediscreteness_inv = -1*sum(particlediscreteness_inv)/float(len(particlediscreteness_inv))

                #If more overall area is attributed to particles in the inverted form, that version is passed to 
                #the calculation steps, both images get written out.

                print sum(arealist_inv), sum(arealist)
                print mean_particlediscreteness_inv, mean_particlediscreteness
                if sum(arealist_inv) > sum(arealist) and mean_particlediscreteness_inv > mean_particlediscreteness:
                    filteredvertices = filteredvertices_inverted
                    inverted = True

    writeout_image(img, filteredvertices, imgname, inverted)

    return filteredvertices, scale, inverted

def after_detection(imgname, filteredvertices, scale, inverted):
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
    # avgarea = avgarea * (scale ** 2)
    #arealist = [a*(scale**2) for a in arealist]
    filtered_areas = remove_outliers(arealist)

    if len(filtered_areas) > 1:
        avgarea = np.median(filtered_areas) * scale ** 2
    else:
        avgarea = float(filtered_areas[0]) * scale ** 2

    arealist = [a*(scale**2) for a in arealist]
    filtered_areas = [a*(scale**2) for a in filtered_areas]

    particle_size_histogram(arealist, filtered_areas, imgname)

    aspect_ratios_list = aspect_ratios(filteredvertices)

    mean_aspect_ratio = round(sum(aspect_ratios_list)/float(len(aspect_ratios_list)),2)

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
    outfile.write("Representative particle size: " + str(avgarea) + " sqm" + "\n")
    outfile.write("Average aspect ratio: " + str(mean_aspect_ratio) + "\n")


    if inverted == True:
        outfile.write("Image colors were inverted for a more accurate detection." + "\n")

    outfile.close()

    #Calculate rdf.
    xRDF = []
    yRDF = []
    if len(filteredvertices) > 9:
        xRDF, yRDF = calculate_rdf(filteredvertices, rows, cols, scale, increment = 4, progress = True)
        output_rdf(xRDF, yRDF, imgname)

    return

def run(path_to_images, path_to_secondary = None, path_to_already_done = None):
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
        secondary.extend([a.split('/')[-1][4:] for a in glob.glob(path_to_secondary)])
    else:
    	secondary = [a.split('/')[-1] for a in images]

    already_done = []
    if path_to_already_done != None:
        #4 is to ignore "det_" prefix.
        already_done.extend([a.split('/')[-1][4:] for a in glob.glob(path_to_already_done)])


    for imgname in images:
        if (imgname.split('/')[-1] in secondary) and (imgname.split('/')[-1] not in already_done) :
            print("Scale and particle detection begun on: " + str(imgname))

            filteredvertices, scale, inverted = main_detection(imgname)

            if len(filteredvertices) > 0:
                after_detection(imgname, filteredvertices, scale, inverted)

            else:
                outfile = open(imgname.split('/')[-1].split(".")[0] + ".txt", "w")
                outfile.write(imgname.split('/')[-1] + "\n")
                outfile.write("No particles found.")
                outfile.close()
            

    return


path_to_images = "/Users/karim/Desktop/evaluation_images/merged/2_karim_split/0_C6CE01551D_fig1_2.png"

path_to_secondary = None

path_to_already_done = None


# path_to_images = "/home/batuhan/Documents/PhD Physics/Projects/imagedataextractor130219_2/merged/2_karim_split/*.png"

# path_to_secondary = "/home/batuhan/Documents/PhD Physics/Projects/imagedataextractor130219_2/merged/4.1_det/*.png"

run(path_to_images, path_to_secondary, path_to_already_done)






