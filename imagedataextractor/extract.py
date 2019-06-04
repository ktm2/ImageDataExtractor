#Inputs 2D image of particles, outputs radial distribution function of particles.
#Karim Mukaddem, Molecular Engineering Group, Department of Physics, University of Cambridge.

#NOTES:
# Some inconsistenies exist in [0],[1]/ x,y indexing of pixels, due to differences in how functions
# in different libraries behave.
# Changing all pixel acessing from a np based approach to cv2's built in (img.item.set) will greatly
# increase efficiency.

# TODO:
# Dark particles on light backgrounds?

import glob
import datetime
import zipfile
import tarfile

from .scalebar_identification import *
from .particle_identification import *
from .rdf_functions import *
from .image_identification import TEMImageExtractor
from .figure_splitting import split_by_photo, split_by_grid


def main_detection(imgname, outputpath=''):
    '''Where the detection happens.

    :param sting imgname: name of image file.
    :param string outputpath: path to output directory.

    :return list filteredvertices: list of vertices of particles in image.
    :return float scale: Scale of pixels in image (m/pixel).
    :return float conversion: unit of scalevalue 10e-6 for um, 10e-9 for nm.

    '''

    img = cv2.imread(imgname)


    if img.shape[0] * img.shape[1] < 50000:
        return None,None,None,None

    scale, inlaycoords, conversion = scalebar_identification(img, outputpath, testing = imgname)

    filteredvertices, particlediscreteness = particle_identification(img, inlaycoords, testing = False)

    inverted = False

    #This is disabled until further development.
    #If less than 3 particles are found, redo analysis with inverted colors.
    # if inverted is False:
    #     rows = len(img)
    #     cols = len(img[0])

    #     if len(img.shape) == 2:
    #         gimg = img
    #     else:
    #         gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    #     if len(filteredvertices) > 0:
    #         arealist = particle_metrics_from_vertices(img, gimg, rows, cols, filteredvertices)[1]
    #         detection_ratio = sum(arealist)/float(rows*cols)
    #         mean_particlediscreteness = sum(particlediscreteness)/float(len(particlediscreteness))
    #     else:
    #         detection_ratio = 0
    #         mean_particlediscreteness = 0
    #         arealist = []

    #     if len(filteredvertices) < 3 or detection_ratio < 0.1 or mean_particlediscreteness < 30:
    #         filteredvertices_inverted, particlediscreteness_inv = particle_identification(img, inlaycoords, testing = True, invert = True)

    #         if len(filteredvertices_inverted) > 0:
    #             if len(img.shape) == 2:
    #                 gimg = img
    #             else:
    #                 gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #             rows = len(img)
    #             cols = len(img[0])

    #             arealist_inv = particle_metrics_from_vertices(img, gimg, rows, cols, filteredvertices_inverted)[1]
    #             mean_particlediscreteness_inv = -1*sum(particlediscreteness_inv)/float(len(particlediscreteness_inv))

    #             #If more overall area is attributed to particles in the inverted form, that version is passed to 
    #             #the calculation steps, both images get written out.

    #             print(sum(arealist_inv), sum(arealist))
    #             print(mean_particlediscreteness_inv, mean_particlediscreteness)
    #             if sum(arealist_inv) > sum(arealist) and mean_particlediscreteness_inv > mean_particlediscreteness:
    #                 filteredvertices = filteredvertices_inverted
    #                 inverted = True

    writeout_image(img, outputpath, filteredvertices, imgname, inverted)

    return filteredvertices, scale, inverted, conversion

def after_detection(imgname, filteredvertices, scale, inverted, conversion, outputpath=''):
    '''After detection has happened calculate particle metrics and RDF.

    :param sting imgname: name of image file.
    :param list filteredvertices: list of vertices of particles (as numpy.ndarray) in image.
    :param float scale: Scale of pixels in image (m/pixel).
    :param float conversion: unit of scalevalue 10e-6 for um, 10e-9 for nm.
    :param string outputpath: path to output directory.


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

    resemblances, conclusion = match_to_shapes(filteredvertices, image_with_shapes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes_to_match.png"))

    #Calculate particle metrics.
    colorlist, arealist, avgcolormean, avgcolorstdev, avgarea = particle_metrics_from_vertices(img, gimg, rows,
        cols, filteredvertices)

    filtered_areas = remove_outliers(arealist)

    if len(filtered_areas) > 1:
        avgarea = np.median(filtered_areas) * scale ** 2
    else:
        avgarea = float(filtered_areas[0]) * scale ** 2

    #Correct sig figs for avgarea
    om = int(math.floor(math.log10(avgarea)))

    avgarea = round(avgarea * 10 ** (-1 * om), 2) * 10 ** (om)




    #Convert from pixel square to meter square.
    arealist = [a*(scale**2) for a in arealist]
    filtered_areas = [a*(scale**2) for a in filtered_areas]

    if len(arealist) > 1 and len(filtered_areas) > 1:
        particle_size_histogram(arealist, filtered_areas, imgname, outputpath, conversion)

    aspect_ratios_list = aspect_ratios(filteredvertices)

    mean_aspect_ratio = round(sum(aspect_ratios_list)/float(len(aspect_ratios_list)),2)

    #number of particles.
    number_of_particles = len(filteredvertices)

    #Log for file.
    outfile = open(os.path.join(outputpath, imgname.split('/')[-1].split(".")[0] + ".txt"), "w")

    outfile.write(imgname.split('/')[-1] + " processed using ImageDataExtractor on "+"\n")
    outfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n")
    outfile.write("Article DOI: " + "\n")
    outfile.write(imgname.split('/')[-1].split("_")[1] + "\n")
    outfile.write("Figure number: " + "\n")
    outfile.write(imgname.split('/')[-1].split("_")[2] + "\n" + "\n")
    # outfile.write("Particle Number, Size in Pixels" + "\n")
    # particle_index = 1
    # for area in arealist:
    #     outfile.write(str(particle_index) + ", " + str(area) + "\n")
    #     particle_index+=1
    outfile.write(str(number_of_particles) + " particle(s) detected." + "\n")
    if conversion == 1:
        outfile.write("Representative particle size: " + str(avgarea) + " sqpx" + "\n" + "\n")
    else:
        outfile.write("Representative particle size: " + str(avgarea) + " sqm" + "\n" + "\n")

    outfile.write("Particle resemblances to regular shapes: " + "\n")
    for i in resemblances:
        outfile.write(str(i) + " ")
    outfile.write("\n")
    outfile.write(conclusion + "\n")
    outfile.write("Average aspect ratio: " + str(mean_aspect_ratio) + "\n")

    if conversion == 1:
        outfile.write("***All units are in pixels as scale of image could not be determined." + "\n")



    if inverted == True:
        outfile.write("Image colors were inverted for a more accurate detection." + "\n")

    outfile.close()

    #Calculate rdf.
    xRDF = []
    yRDF = []
    if len(filteredvertices) > 9:
        xRDF, yRDF = calculate_rdf(filteredvertices, rows, cols, scale, increment = 4, progress = True)
        output_rdf(xRDF, yRDF, imgname, conversion, outputpath)

    return


def extract_images(path_to_images, outputpath='', path_to_secondary = None, path_to_already_done = None):
    '''Runs scalebar and particle identification on an image document.

    :param string path_to_images: path to images of interest.
    :param string outputpath: path to output directory.
    :param string path_to_secondary: path to secondary directory, useful
    to skip certain images (if starting or restarting a batch) or process only
    a preapproved set. 

    '''

    if os.path.isdir(path_to_images):
        images = [os.path.join(path_to_images, img) for img in os.listdir(path_to_images)]
    elif os.path.isfile(path_to_images):

        # Unzipping compressed inputs
        if path_to_images.endswith('zip'):
            # Logic to unzip the file locally
            print('Opening zip file...')
            zip_ref = zipfile.ZipFile(path_to_images)
            extracted_path = os.path.join(os.path.dirname(path_to_images), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            zip_ref.extractall(extracted_path)
            zip_ref.close()
            images = [os.path.join(extracted_path, img) for img in os.listdir(extracted_path)]

        elif path_to_images.endswith('tar.gz'):
            # Logic to unzip tarball locally
            print('Opening tarball file...')
            tar_ref = tarfile.open(path_to_images, 'r:gz')
            extracted_path = os.path.join(os.path.dirname(path_to_images), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            tar_ref.extractall(extracted_path)
            tar_ref.close()
            images = [os.path.join(extracted_path, img) for img in os.listdir(extracted_path)]

        elif path_to_images.endswith('tar'):
            # Logic to unzip tarball locally
            print('Opening tarball file...')
            tar_ref = tarfile.open(path_to_images, 'r:')
            extracted_path = os.path.join(os.path.dirname(path_to_images), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            tar_ref.extractall(extracted_path)
            tar_ref.close()
            images = [os.path.join(extracted_path, img) for img in os.listdir(extracted_path)]
        else:
            images = [path_to_images]
    else:
        raise Exception('Unsupported input format')

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

            imgoutputdir = os.path.join(outputpath, imgname.split('/')[-1].split('.')[0])
            if not os.path.exists(imgoutputdir):
                os.makedirs(imgoutputdir)

            filteredvertices, scale, inverted, conversion = main_detection(imgname, imgoutputdir)

            if filteredvertices == None:
                
                outfile = open(os.path.join(imgoutputdir, imgname.split('/')[-1].split(".")[0] + ".txt"), "w")
                outfile.write(imgname.split('/')[-1] + " processed using ImageDataExtractor on "+"\n")
                outfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n")
                outfile.write("Article DOI: " + "\n")
                outfile.write(imgname.split('/')[-1].split("_")[1] + "\n")
                outfile.write("Figure number: " + "\n")
                outfile.write(imgname.split('/')[-1].split("_")[2] + "\n" + "\n")
                outfile.write("Image does not meet resolution requirements.")
                outfile.close()


            elif len(filteredvertices) > 0:

                after_detection(imgname, filteredvertices, scale, inverted, conversion, imgoutputdir)

            else:

                outfile = open(os.path.join(imgoutputdir, imgname.split('/')[-1].split(".")[0] + ".txt"), "w")
                outfile.write(imgname.split('/')[-1] + " processed using ImageDataExtractor on "+"\n")
                outfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n")
                outfile.write("Article DOI: " + "\n")
                outfile.write(imgname.split('/')[-1].split("_")[1] + "\n")
                outfile.write("Figure number: " + "\n")
                outfile.write(imgname.split('/')[-1].split("_")[2] + "\n" + "\n")
                outfile.write("No particles found.")
                outfile.close()

    return


extract_image = extract_images


def extract_documents(path_to_documents, path_to_images='', outputpath='', path_to_secondary = None, path_to_already_done = None):
    """ Automatically detects SEM and TEM images from HTML/XML documents for extraction

    :param string path_to_documents : path to documents of interest
    :param string outputpath: path to output directory.
    :param string path_to_secondary: path to secondary directory, useful
    to skip certain images (if starting or restarting a batch) or process only
    a preapproved set.
    """

    extractor = TEMImageExtractor(path_to_documents, path_to_images, typ='tem')
    extractor.get_all_tem_imgs(parallel=False)

    # Split raw images using 2-step splitting pipeline
    split_figures(path_to_images)

    # Extract all split images
    path_to_split_images = os.path.join(path_to_images, 'split_grid_images')
    extract_images(path_to_split_images, outputpath)


def extract_document(path_to_document, path_to_images='', outputpath=''):
    """ Automatically detects SEM and TEM images for a simple HTML/XML document"""

    extractor = TEMImageExtractor(path_to_document, path_to_images, typ='tem')
    extractor.get_tem_imgs()

    # Split raw images using 2-step splitting pipeline
    split_figures(path_to_images)

    # Extract all split images
    path_to_split_images = os.path.join(path_to_images, 'split_grid_images')
    extract_images(path_to_split_images, outputpath)


def split_figures(input_dir, output_dir=''):
    """ Automatically splits hybrid images through photo detection and grid splitting"""

    # Define all used paths
    input_dir_name = os.path.basename(input_dir)
    raw_imgs_path = os.path.join(input_dir, 'raw_images')
    raw_csv_path = os.path.join(input_dir, input_dir_name + '_raw.csv')
    split_photo_imgs_path = os.path.join(input_dir, 'split_photo_images')
    split_photo_csv_path = os.path.join(input_dir, input_dir_name + '_photo.csv')
    split_grid_imgs_path = os.path.join(input_dir, 'split_grid_images')

    # Split using photo detection method
    split_by_photo(raw_imgs_path, raw_csv_path, split_photo_imgs_path, split_photo_csv_path, True)

    # Split output using grid detection method
    split_by_grid(split_photo_imgs_path, split_grid_imgs_path)





#
# path_to_images = "/Users/karim/Desktop/evaluation_images/merged/2_karim_split/0_C6CE01551D_fig1_2.png"
#
# path_to_secondary = "/Users/karim/Desktop/evaluation_images/merged/4.2_det/*.png"
#
# path_to_already_done = None


# path_to_images = "/home/batuhan/Documents/PhD Physics/Projects/imagedataextractor130219_2/merged/2_karim_split/*.png"

# path_to_secondary = "/home/batuhan/Documents/PhD Physics/Projects/imagedataextractor130219_2/merged/4.1_det/*.png"

# Ed's test paths
# path_to_image = "/home/edward/Documents/ImageDataExtractor_documents/test_cde_ide/output/split_grid_images/0_C6CE01551D_fig1_2.png"
#path_to_documents = "/home/edward/Documents/ImageDataExtractor_documents/test_cde_ide/input"
# path_to_image_output = "/home/edward/Documents/ImageDataExtractor_documents/test_cde_ide/output"
# path_to_ide_output = "/home/edward/Documents/ImageDataExtractor_documents/test_cde_ide/ide_output"
#
# #output_path = "/home/edward/Pictures/ImageDataExtractor_images/output/"
#
#
# #extract_image(path_to_image)
# #extract_documents(path_to_documents, path_to_image_output, path_to_ide_output)
#extract_document(os.path.join(path_to_documents, 'C6CE01551D.html'))#, path_to_image_output, path_to_ide_output)
#
# input_dir = '/home/edward/Documents/ImageDataExtractor_documents/test_cde_ide/output'
#
# #split_figures(input_dir)






