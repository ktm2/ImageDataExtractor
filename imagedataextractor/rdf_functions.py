
import matplotlib.pyplot as plt
import math

from .img_utils import *


def calculate_rdf(filteredvertices,rows,cols,scale, increment = 4, progress = False):
    '''Calculates RDF from list of vertices of particles.

    :param list filteredvertices: list of vertices of particles.
    :param int rows: number of rows in image.
    :param int cols: number of cols in image.
    :param float scale: scale of pixels in image (m/pixel)
    :param int increment: Increment resolution of RDF in pixels.
    :param bool progress: Optionally print progress.

    :return list xRDF: x values of RDF
    :return list yRDF: y values of RDF

    TODO:
    - Must be better way of doing this than pixel counting?

    '''


    if progress == True:
        print("RDF calculation: ")


    #Create blank canvases to draw particle pairs on.
    particleAimg = np.zeros((rows, cols, 3), np.uint8)
    particleAimg[:] = (0, 0, 0)

    particleBimg = np.zeros((rows, cols, 3), np.uint8)
    particleBimg[:] = (0, 0, 0)


    #Calculation of RDF.
    minrange = 1
    maxrange = int(((rows**2)+(cols**2))**0.5)
    xRDF = range(minrange,maxrange,increment)


    #Start with no intersects at any radius.
    AllIntersectsAtRadius=[]
    for i in xRDF:
        AllIntersectsAtRadius.append(0)

    numberofparticles = len(filteredvertices)

    particle_index = 1
    
    #Calculate all particleA to particleB pairs

    for particleA in filteredvertices:

        if progress == True:
            print('RDF calculation on ' + str(particle_index) + '/' + str(numberofparticles))
            particle_index += 1

        restofvertices=[particleB for particleB in filteredvertices
            if np.array_equal(particleA,particleB) == False]

        (particleAx,particleAy),particleAradius = cv2.minEnclosingCircle(particleA)

        particleAx = int(particleAx)
        particleAy = int(particleAy)
        particleAradius = int(particleAradius)

        cv2.polylines(particleAimg,[particleA],True,(0,0,255))

        for particleB in restofvertices:
            
            #paint blank particleBimg with a filled particleB.
            
            cv2.polylines(particleBimg,[particleB],True,(0,255,0)) #might be unnecessary
            cv2.fillPoly(particleBimg,[particleB],(0,255,0))

            (particleBx,particleBy),particleBradius = cv2.minEnclosingCircle(particleB)
            particleBx=int(particleBx)
            particleBy=int(particleBy)
            particleBradius=int(particleBradius)
            
            ABintersects=[]

            min_distance_for_this_pair = (distance_formula((particleAx,particleAy),(particleBx,particleBy))
             - int(particleBradius) * 1.5)

            max_distance_for_this_pair = (distance_formula((particleAx,particleAy),(particleBx,particleBy))
             + int(particleBradius) * 1.5)

            for i in xRDF:

                doesABintersectAtRadius=0

                if  min_distance_for_this_pair < i < max_distance_for_this_pair:

                    #Draw a circle of radius i originating from center of particleA.

                    cv2.circle(particleAimg,(particleAx,particleAy),i,(255,0,0))


                    #Combine the two images.

                    combinedimg=cv2.addWeighted(particleBimg,1,particleAimg,1,0)

                    krange=range((particleBx-particleBradius),(particleBx+particleBradius))
                    krangetrim=[k for k in krange if k < cols]

                    lrange=range((particleBy-particleBradius),(particleBy+particleBradius))
                    lrangetrim=[l for l in lrange if l < rows]

                    #Search general region of particle B to see if the circle of radius i shows up in it.
                    #If so, there is an intersect between particles A & B at radius i. 

                    for k in krangetrim:
                            for l in lrangetrim:
                                if (combinedimg.item(l,k,0) == 255 and combinedimg.item(l,k,1) == 255):
                                    doesABintersectAtRadius = 1                                
                                    break

                ABintersects.append(doesABintersectAtRadius)

                particleAimg[:] = (0, 0, 0)             

            particleBimg[:] = (0, 0, 0)    

            AllIntersectsAtRadius=[x + y for x, y in zip(AllIntersectsAtRadius, ABintersects)]

            
    yRDF=[i/float(numberofparticles) for i in AllIntersectsAtRadius]

    #Convert pixels to unit of distance.
    xRDF=[x*scale for x in xRDF]

    return xRDF,yRDF



def output_rdf(xRDF,yRDF,imgname,conversion, outputpath=''):
    '''Plots a given rdf.

    :param string outputpath: path to output directory.
'''
    
    if conversion == 1:
        distanceunit = 'pixels'
    else:
        om = int(math.floor(math.log10(conversion)))
        distanceunit= 'meters E' + str(om)

    xRDF = [round(i * 10 ** (-1*om),2) for i in xRDF]

    plt.plot(xRDF,yRDF, label="_nolegend_",marker='o', linestyle = "None")
    font={"fontname":"serif"}
    plt.ylim([0,max(yRDF)+(max(yRDF)/10.0)])
    plt.xlim([0,max(xRDF)])
    plt.title("minRDF",**font)
    plt.xlabel("distance / " + distanceunit,**font)
    plt.ylabel("Frequency",**font)
    plt.grid()

    plt.savefig(os.path.join(outputpath, "rdf_" + str(imgname).split("/")[-1]), bbox_inches = 'tight')
    plt.close()
    #plt.show()

    outfile = open(os.path.join(outputpath, "data_rdf_" + imgname.split('/')[-1].split(".")[0] + ".txt"), "w")

    for i,j in zip(xRDF,yRDF):
        outfile.write(str(i) + " " + str(j) + "\n")

    outfile.close()


    return

def particle_size_histogram(arealist, filtered, imgname, outputpath='', conversion = 0):
    '''Plots particle size histogram.
    :param list arealist: list of the areas of particles.
    :param string imgname: name of the img (needed for writing output)
    :param string outputpath: path to output directory.
    :param float conversion: order of magnitude of scale in image, if 1 scale is in pixels.
    '''

    font={"fontname":"serif"}
    _, bins, _ = plt.hist(arealist, bins=len(arealist) + 1, edgecolor='black', linewidth=1.2, rwidth=0.9, label='Original')
    plt.hist(filtered, bins=len(arealist) + 1, range=(bins.min(), bins.max()), edgecolor='black', linewidth=1.2, 
        rwidth=0.9, label='Filtered', alpha = 0.6)
    plt.title("Particle Size " + str(imgname).split("/")[-1] ,**font)
    if conversion == 1:
        plt.xlabel('Pixels**2',**font)
    else:
        plt.xlabel('Meters**2',**font)
    plt.ylabel("Frequency",**font)
    ##is this right?
    plt.xlim([0,max(arealist)])
    plt.legend()
    plt.savefig(os.path.join(outputpath, "hist_" + str(imgname).split("/")[-1]), bbox_inches = 'tight')
    plt.close()
    #plt.show()

    return


def aspect_ratios(filteredvertices):
    '''Calculates aspect ratios of particles.
    :param list filteredvertices: list of detected particles.

    :return list aspect_ratios: list of respective aspect ratios.'''

    aspect_ratios = []

    for cont in filteredvertices:
        x,y,w,h = cv2.boundingRect(cont)
        aspect_ratio = float(w)/h
        aspect_ratios.append(aspect_ratio)

    return aspect_ratios

def remove_outliers(areas):
    if len(areas) > 1:
        mu = np.median(areas)
        std = np.std(areas)
        filtered = [x for x in areas if (x < mu + (1.75*std) and x > mu - (1*std))]
    else:
        filtered = areas
    return filtered

