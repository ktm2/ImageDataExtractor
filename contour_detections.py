import cv2
import numpy as np
from img_utils import *
from scipy.stats import mode


def match_to_shapes(filteredvertices, image_with_shapes = "shapes_to_match.png"):
    '''Determine closeness of shapes to shapes in a given example image, using Hu-moments.
    Mostly orientation and scale invariant.
    :param numpy.ndarray filteredvertices: the detected particles from earlier in detection.
    :param string image_with_shapes: name of the input image with target shapes.

    :return list 
    :return string
    '''

    shapes_to_match = cv2.imread(image_with_shapes)
    shape_vertices = find_draw_contours(shapes_to_match)

    #Have to be provided by user if a custom image is being used
    #suggest using testing and annotation mode in find_draw_contours to do this.
    shape_labels = ["circle","ellipse","square","rectangle"]

    #List of length of number of particles
    #Each element is a list of length 4, with the corresponding matching coefficients
    #to the regular shapes in shape_labels. Closer to 0 means closer match.
    match = []
    for particle in filteredvertices:
        particle_matches = []
        for shape in shape_vertices:
            ret = cv2.matchShapes(np.array(particle),np.array(shape[0]),1,0.0)
            particle_matches.append(round(ret,2))
        match.append(particle_matches)


    modes = [mode(i)[0] for i in zip(*match)]
    
    #Mean like approach.
    #overall_matches = [round(sum(i) / float(len(filteredvertices)),2) for i in zip(*match)]


    if min(modes) < 0.1:
        conc = str("The 2D projections of the particles in this image most closely match a: " 
        + shape_labels[modes.index(min(modes))])
    else:
        conc = str("The geometry 2D projections of the particles in this image cannot be classified.")


    return zip(shape_labels,[str(i[0]) for i in modes]), conc



def draw_contours(gimg,filteredvertices,imgname="img"):
    '''Draw detected particles on an image.
    :param numpy.ndarray gimg: original image in grayscale
    :param list filteredvertices: list of vertices of detected particles
    :param string imgname: header while displaying image.
    '''

    #img in grayscale,convert to color.
    if len(gimg.shape)==2:
        backtorgb = cv2.cvtColor(gimg,cv2.COLOR_GRAY2RGB)

    #img is in already in color.
    else:
        backtorgb=gimg

    for vertices in filteredvertices:
        cv2.polylines(backtorgb,[vertices],True,(0,0,255),thickness=1)

    show_image(backtorgb,waitkey=0,imgname=imgname)

    return None


def find_draw_contours(img, blocksize = 151, blursize = 0, minarea = None, 
    maxarea=None, testing = False, annotate=False, bounding_rect=False,
    nofilter=False, restrict_rectangles=False, restrict_rectangles_color=None, contours_color = (0,0,255)):
    '''Find and draw contours on a given image.
    :param numpy.ndarray img: input image
    :param int blocksize: associated with adaptive thresholding.
    :param int blursize: associated with blurring step for noise reduction.
    :param int minarea: Minimum contour size in square pixels.
    :param int maxarea: Maximum contour size in sqaure pixels.
    :param bool testing: Display image.
    :param bool annotate: Annotate the found particles in the output image.
    :param bool bounding_rect: Calculate the bounding rectangles of the found particles.
    :param bool nofilter: If true skip filtering and search raw image for contours.
    :param restrict_rectangles: Apply dimensional restrictions before outputting bounding rectangles.
    :param restrict_rectangles_color: Apply color restrictions before outputting bounding rectangles.
    :param tuple contours_color: (b,g,r) of what color to draw detected contours if testing = True.


    :return list filteredvertices:

    if bounding_rect is True additionally:

    :return list boundingrectanglewidths:
    :return list boundingrectanglecoords:

    TODO:
    - This function and find_draw_contours are very similar but both bulky, 
    both should be broken apart and rebuilt more efficiently.
    - If statement on returns is not ok?

    '''
 
    imgarea=len(img)*len(img[0])

    #Apply filters to image.
    #img already grayscale.
    if len(img.shape)==2:
        gimg=img
        drawingimg=cv2.cvtColor(gimg,cv2.COLOR_GRAY2RGB)
        outputimg=img.copy()
    #img is in color, convert to grayscale.
    else:
        gimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        drawingimg=img
        outputimg=img.copy()


    if nofilter == False:
        thresh1 = cv2.adaptiveThreshold(gimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=blocksize,C=0)
    else:
        thresh1 = gimg

    #Apply blur filter (optional), find contours.
    if blursize!=0:
        thresh=cv2.medianBlur(thresh1,blursize)
        unknownvar,contours,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        unknownvar,contours,h = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)




    #Set variables for contour processing.
    shapeindex=1
    arealist=[]
    filteredcontours=[]
    filteredvertices=[]
    shapecoords=[]
    contindex=0
    boundingrectanglewidths=[]
    boundingrectanglecoords=[]

    if minarea==None:
        minarea=imgarea*0.001

    if maxarea==None:
        maxarea=imgarea*0.5




    #Contour processing.
    for cont in contours:
        #Calculate contour areas, fit polygons to contours and establish hierarchical relations (useful for processing nested contours).
        area=cv2.contourArea(cont)
        vertices=(cv2.approxPolyDP(cont,0.0025*cv2.arcLength(cont,True),True)) #0.0025 determines strictness of polygon fitting.
        parent=h[0][contindex][3]

        #Filter out contours using area boundaries and hierarchical relations.
        if area>minarea and area<maxarea and len(vertices)>2 and (parent==0 or parent==-1):


            filteredcontours.append(cont)
            filteredvertices.append([vertices])
            arealist.append(area)


            #Draw filtered contours on img.
            cv2.polylines(drawingimg,[vertices],True,contours_color,thickness=1)


            if bounding_rect==True:
                #x,y are top left coordinates of bounding rectangle.
                boundingx,boundingy,width,height=cv2.boundingRect(vertices)

                #print area,0.5*width*height

                if restrict_rectangles==True and restrict_rectangles_color == False:


                    if area>0.4*width*height: #boundingy!=0 and boundingx!=0 and a:
                        boundingrectanglewidths.append(width)
                        boundingrectanglecoords.append((boundingx,boundingy,width,height))

                elif restrict_rectangles == True and restrict_rectangles_color != None:

                    rect_std = np.std(gimg[boundingy:boundingy+height,boundingx:boundingx+width].flatten())
                    rect_mean = np.mean(gimg[boundingy:boundingy+height,boundingx:boundingx+width].flatten())

                    #show_image(gimg[boundingy:boundingy+height,boundingx:boundingx+width],0,str(int(rect_std))+ " " + str(int(rect_mean)))

                    # print area,width,height,rect_std,rect_mean
                    # print restrict_rectangles_color

                    if restrict_rectangles_color == "white":
                        if area>0.4*width*height and rect_std < 50 and rect_mean > 150: #boundingy!=0 and boundingx!=0 and a:
                            boundingrectanglewidths.append(width)
                            boundingrectanglecoords.append((boundingx,boundingy,width,height))  

                    elif restrict_rectangles_color == "black":
                        if area>0.4*width*height and rect_std < 50 and rect_mean < 30: #boundingy!=0 and boundingx!=0 and a:
                            boundingrectanglewidths.append(width)
                            boundingrectanglecoords.append((boundingx,boundingy,width,height))                                           

                else:
                    boundingrectanglewidths.append(width)
                    boundingrectanglecoords.append((boundingx,boundingy,width,height))


                if annotate==True:
                    cv2.rectangle(drawingimg,(boundingx,boundingy),(boundingx+width,boundingy+height),(0,0,255),thickness=1)



            if annotate==True:
                #Annotate and number centers of filtered contours.
                (xcom,ycom),contradius = cv2.minEnclosingCircle(vertices)
                xcom=int(xcom)
                ycom=int(ycom)
                contradius=int(contradius)
                shapecoords.append((xcom,ycom,contradius))
                cv2.circle(drawingimg,(xcom,ycom),1,(0,0,255),1)
                cv2.putText(drawingimg,str(shapeindex),(xcom+3,ycom+3),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,0,255),thickness=1)           
                shapeindex+=1




        contindex+=1
        
    #Displays output image.
    if testing == True:
        show_image(drawingimg,0,imgname="find_draw_contours")

    if bounding_rect==True:
        return filteredvertices,boundingrectanglewidths,boundingrectanglecoords
    else:
        return filteredvertices


def find_draw_contours_main(img, gimg,blocksize, rows, cols, blursize = 0, testing = False,
    annotate = False, unbounded = False):
    '''Find and draw contours on a given image, this functions is similar to find_draw_contours, but has less options
    and is used for the initial contour detection.

    :param numpy.ndarray img: input image
    :param numpy.ndarray gimg: input image in grayscale.

    :param int blocksize: associated with adaptive thresholding.
    :param int rows: number of rows in image.
    :param int cols: number of cols in image.
    :param int blursize: associated with blurring step for noise reduction.
    :param bool testing: Display image.
    :param bool annotate: Annotate the found particles in the output image.
    :param bool unbounded: Apply size restrictions to contours or not.

    :return list filteredvertices:

    TODO:
    - This function and find_draw_contours are very similar but both bulky, 
    both should be broken apart and rebuilt more efficiently.

    '''

    outputimg = img.copy()
    imgarea = rows*cols



    thresh1 = cv2.adaptiveThreshold(gimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=blocksize,C=0)


    #Apply blur filter (optional), find contours.
    if blursize!=0:
        thresh=cv2.medianBlur(thresh1,blursize)
        unknownvar,contours,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        unknownvar,contours,h = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)



    #Set variables for contour processing.
    shapeindex=1
    filteredcontours=[]
    filteredvertices=[]
    contindex=0

    if unbounded==True:
        minarea=0
        maxarea=imgarea
    else:
        minarea=imgarea*0.001
        maxarea=imgarea*0.5

    #Contour processing.
    for cont in contours:
        #Calculate contour areas, fit polygons to contours and establish hierarchical relations (useful for processing nested contours).
        area=cv2.contourArea(cont)
        vertices=(cv2.approxPolyDP(cont,0.0025*cv2.arcLength(cont,True),True)) #0.0025 determines strictness of polygon fitting.
        parent=h[0][contindex][3]

       


        #Filter out contours using area boundaries and hierarchical relations.
        if area>minarea and area<maxarea and len(vertices)>2 and (parent==0 or parent==-1):

            filteredcontours.append(cont)
            filteredvertices.append(vertices)

            #Draw filtered contours on img.
            cv2.polylines(outputimg,[vertices],True,(0,255,0),thickness=1)


            #Annotate and number centers of filtered contours.
            if annotate==True:
                (xcom,ycom),contradius = cv2.minEnclosingCircle(vertices)
                xcom=int(xcom)
                ycom=int(ycom)
                contradius=int(contradius)
                cv2.circle(outputimg,(xcom,ycom),1,(0,0,255),1)
                cv2.putText(outputimg,str(shapeindex),(xcom+3,ycom+3),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,0,0),thickness=1)           
                shapeindex+=1





        contindex+=1
        
    #Displays output image.
    if testing == True:
        show_image(outputimg,0,imgname="all contours")


    return filteredvertices
