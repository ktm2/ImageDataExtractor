import cv2
import numpy as np
import itertools
import math
from contour_detections import *

def edge_correction(filteredvertices,rows,cols,inlaycoords, testing = False, gimg = None):
    '''Filters out particles that are deformed by image borders or inlays. (must intersect either at min 2 points)

    :param list filteredvertices: list of vertices of deteced particles.
    :param int rows: number of rows in image.
    :param int cols: number of columns in image.
    :param list inlaycoords: list of coordinates (as tuples) of inlays in image, including text info.
    :param bool testing: show what's removed or not.
    :param numpy.ndarray gimg: input image, mandatory if testing is True.

    :return list edgecorrectedvertices: list of vertices of filtered particles. 
    '''

    edgecorrectedvertices=[]

    removedvertices = []

    if testing ==True:
        test_img = cv2.cvtColor(gimg,cv2.COLOR_GRAY2RGB)

    for vertices in filteredvertices:
        borderintersections=0
        inlayintersections=0


        for point in vertices:

            if len(point) < 2:
                point = point[0]
            
            #image borders.
            if (point[0] >= cols-1) or (point[1] >= rows-1) or (0 in point):
                borderintersections+=1

            #inlay borders.
            if len(inlaycoords) > 0:
                for inlay in inlaycoords:
                    #Grow inlays slightly, to account for points very close.
                    growby = 2 #pixels

                    inlaycorners=((inlay[0]-growby,inlay[1]-growby),(inlay[0]+inlay[2]+growby,inlay[1]-growby)\
                         ,(inlay[0]+inlay[2]+growby,inlay[1]+inlay[3]+growby),(inlay[0]-growby,inlay[1]+inlay[3]+growby))
                    #is point in or on the edge of any inlays.
                    if (point[0]>=inlaycorners[0][0] and point[0]<=inlaycorners[1][0] and point[1]>=inlaycorners[0][1]\
                        and point[1]<=inlaycorners[3][1]):
                        inlayintersections+=1


        if (borderintersections<2) and (inlayintersections<2):
            edgecorrectedvertices.append(vertices)
        else:
            removedvertices.append(vertices)

    if testing == True:
        print(str(len(removedvertices)) + " particles removed.")

        for inlay in inlaycoords:
            growby = 2 #pixels
            inlaycorners=((inlay[0]-growby,inlay[1]-growby),(inlay[0]+inlay[2]+growby,inlay[1]-growby)\
                         ,(inlay[0]+inlay[2]+growby,inlay[1]+inlay[3]+growby),(inlay[0]-growby,inlay[1]+inlay[3]+growby))

            cv2.rectangle(test_img,inlaycorners[0], inlaycorners[2],(255,0,0),thickness = 1)


        draw_contours(test_img,removedvertices,imgname = "edge removed particles")




    return edgecorrectedvertices


def particle_metrics_from_vertices(img,gimg,rows,cols,filteredvertices, invert = False):
    '''Returns particle metrics using vertices and img
    :param numpy.ndarray img: original RGB image
    :param numpy.ndarray gimg: original img in grayscale
    :param int rows: number of rows in image.
    :param int cols: number of columns in image.
    :param list filteredvertices: list of vertices of detected particles.
    :param bool invert: If an inverted image is being used, makes sure the original is used for metrics.

    :return list colorlist: list of average color (mean pixel intensity) of each particle.
    :return list arealist: list of average size (how many pixels) of each particle.
    :return float avgcolormean: global color average
    :return float avgcolorstdev: average standard deviation of pixel intensities within particles (how
    	complexly colored are the particles)
    :return float avgarea: average size of particles

    '''

    if invert == True:
        gimg = (255-gimg)

    arealist = []
    colorlist = []

    areatotal = 0
    colormeantotal = 0
    colorstdevtotal = 0

    for vert in filteredvertices:
        area=cv2.contourArea(vert)

        arealist.append(area)
        areatotal+=area

        indivcolorimg=img.copy()
        cv2.fillPoly(indivcolorimg,[vert],(0,0,255))

        shapearea=[]

        (xcom,ycom),contradius = cv2.minEnclosingCircle(vert)
        xcom=int(xcom)
        ycom=int(ycom)
       	contradius=int(contradius)

        #y and x's are inverted due to the way you scan a np. array..
        ystart=int(xcom-2*contradius) 
        yend=int(xcom+2*contradius)   
        xstart=int(ycom-2*contradius)
        xend=int(ycom+2*contradius)

        if ystart<0:
            ystart=0
        if yend>cols:
            yend=cols
        if xstart<0:
            xstart=0
        if xend>rows:
            xend=rows

        for x in range(xstart,xend):
            for y in range(ystart,yend):
                if indivcolorimg[x][y][2]==255 and indivcolorimg[x][y][0]==0:
                    shapearea.append(gimg[x][y])


        colormean=sum(shapearea)/len(shapearea)
        colorstdev=np.std(shapearea)

        #Append particle color mean and standard deviation.
        colorlist.append((colormean,colorstdev))

        colormeantotal+=colormean
        colorstdevtotal+=colorstdev


    numberofparticles=len(arealist)

    avgcolormean=float(colormeantotal)/numberofparticles
    avgcolorstdev=float(colorstdevtotal)/numberofparticles
    avgarea=float(areatotal)/numberofparticles

    return colorlist,arealist,avgcolormean,avgcolorstdev,avgarea


def false_positive_correction(filteredvertices,arealist,colorlist,avgcolormean,avgcolorstdev,testing = False,gimg=None):
    '''Determines false positives in a list of contour by comparing individual
    color mean and stdev to global color mean and stdev.

    :param list filteredvertices: list of vertices of detected particles.
    :param list arealist: list of sizes of detected particles (in same order).
    :param list colorlist: list of mean pixel intensities of detected particles.
    :param float avgcolormean: mean color of particles.
    :param float avgcolorstdev: average color stdev of particles.
    :param bool testing: Display which particles are being removed. (will require gimg input)
    :param numpy.ndarray gimg: The original image, required if testing is True.

    :return list filteredvertices: corrected list of vertices of particles.
    :return list arealist: updated list of particle sizes.

    TODO:

    - Absolute floor for colorstdev necessary?? check for img det_0_C4CE00151F_fig1_1

    '''

    #Create new list of false-positive-filtered vertices.


    indextoremove=[]

    for i in range(len(colorlist)):
        if ((colorlist[i][0]<avgcolormean*0.6 and colorlist[i][1]<avgcolorstdev*0.25) \
         or colorlist[i][0]<2 or colorlist[i][0] > 240) == True:

        #or colorlist[i][0]>1.6*avgcolormean

            indextoremove.append(i)

    indextoremove=list(set(indextoremove))

    if testing == True:
        print(str(len(indextoremove)) + " false positives removed.")

        if len(indextoremove)>0:
            removedvertices=[]
            for i in indextoremove:
                removedvertices.append(filteredvertices[i])

            draw_contours(gimg,removedvertices,imgname="removed particles")

    for i in sorted(indextoremove,reverse=True):
        del filteredvertices[i]
        del arealist[i]



    return filteredvertices,arealist


def cluster_breakup_correction(filteredvertices, rows, cols, arealist, avgarea, blocksize, testing = False, detailed_testing = False):
    '''Attempt to break apart clusters of particles mistakenly detected as one large particle by identifiying 
    bottle necks (the thinnest part where two particles are "sintered") and joining those bottle necks. 

    :param list filteredvertices: list of vertices of detected particles.
    :param int rows: number of rows in image.
    :param int cols: number of columns in image.
    :param list arealist: list of sizes of detected particles (in same order).
    :param float avgarea: average particle size. 
    :param int blocksize: parameter associated with adaptive filtering, passed down from previous detections.
    :param bool testing: display what's happening step by step.
    :param bool detailed_testing: display what's happening step by step, shows more substeps.


    :return list clusterbreakupvertices: corrected list of vertices of detected particles.

    TODO:
    - Ellipse sizes should not depend on avg_area, maybe on overall img size?

    should pixels_to_extend_by depend on the length of the connection line?    
    '''

    breakuptesting = testing
    
    #Create blank images to work off of. 

    updatingimg=np.zeros((rows,cols,3),np.uint8)
    updatingimg[:]=(0,0,0)

    testimg=np.zeros((rows,cols,3),np.uint8)

    clusterbreakupvertices=list(filteredvertices)
    indextoremove=[]

    for i in range(len(filteredvertices)):

        breakupimg = np.zeros((rows, cols, 3), np.uint8)
        breakupimg[:] = (0, 0, 0)

        maskimg=breakupimg.copy()


        if arealist[i]> 0.5*avgarea or arealist[i]>(rows*cols)/2:


            cv2.polylines(breakupimg,[filteredvertices[i]],True,(0,255,0),thickness=1)
            #cv2.polylines(updatingimg,[filteredvertices[i]],True,(255,255,255),thickness=1)
            #cv2.fillPoly(breakupimg,[filteredvertices[i]],(0,0,255))
            cv2.fillPoly(updatingimg,[filteredvertices[i]],(0,0,255))

            vert = np.array(filteredvertices[i])
            hull = cv2.convexHull(vert,returnPoints = False)
            defects = cv2.convexityDefects(vert,hull)

            if defects is not None:

                for j in range(defects.shape[0]):
                    s,e,f,d = defects[j,0]

                    if len(vert[s])>1:
                        start = tuple(vert[s])
                        end = tuple(vert[e])
                        far = tuple(vert[f])
                    else:
                        start = tuple(vert[s][0])
                        end = tuple(vert[e][0])
                        far = tuple(vert[f][0])

                    if (d/256.0) > (arealist[i]**0.5)/15.0:

                            # print "arearoot/10 ", (arealist[i]**0.5)/10



                            indextoremove.append(i)
                            midpoint=((start[0]+end[0])/2,(start[1]+end[1])/2)

                            #Attempt to slice contour horizontally or vertically in order to find closest point to convexhull point on other side 
                            #of contour to accurately seal the bottleneck. 


                            #Some trigonometry to figure out rotation angle of ellipse.
                            #Tangent Method.
                            
                            #deal with exceptions first.
                            if (midpoint[0]-far[0])==0:
                                if midpoint[1]<far[1]:
                                    angle=0
                                else:
                                    angle=180
                            elif (midpoint[1]-far[1])==0:
                                if far[0]<midpoint[0]:
                                    angle=90
                                else:
                                    angle=-90
                            #calculate angle of non horizontal or non vertical lines.
                            else:
                                gradient=(midpoint[1]-far[1])/float(midpoint[0]-far[0])
                                angle=math.degrees(math.atan(gradient))


                            #Correct for quadrants.
                            if midpoint[0]>far[0] and midpoint[1]>far[1]:
                                angle=90+angle
                            elif midpoint[0]>far[0] and midpoint[1]<far[1]:
                                angle=90+angle
                            elif midpoint[0]<far[0] and midpoint[1]>far[1]:
                                angle=angle-90
                            elif midpoint[0]<far[0] and midpoint[1]<far[1]:
                                angle=angle-90
                            
                            maskimg[:] = (0, 0, 0)
                            l=int((avgarea**0.5)/4)
                            cv2.ellipse(maskimg, far, (int(0.9*l),2*l), angle, 0, 180, (255,255,255), -1)

                            oppositeface=[]

                            maskedresult = np.bitwise_and(breakupimg,maskimg)

                            visualizemasking = detailed_testing
                            
                            if visualizemasking==True:

                                maskingvisualization=breakupimg.copy()
                                # cv2.putText(maskingvisualization,"start",start,cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),thickness=1)  
                                # cv2.putText(maskingvisualization,"end",end,cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),thickness=1) 
                                # #cv2.circle(maskingvisualization,far,1,(0,0,255),1)
                                # cv2.putText(maskingvisualization,"mp",midpoint,cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),thickness=1) 


                                cv2.circle(maskingvisualization,start,2,(255,255,255),-1)
                                cv2.circle(maskingvisualization,end,2,(255,255,255),-1)
                                cv2.circle(maskingvisualization,far,2,(0,0,255),-1)
                                

                                cv2.ellipse(maskingvisualization, far, (int(0.9*l),2*l), angle, 0, 180, (255,255,255), 1)

                                #cv2.putText(maskingvisualization,str(gradient),midpoint,cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),thickness=1) 
                                #cv2.putText(maskingvisualization,str(angle),far,cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),thickness=1)

                                show_image(maskingvisualization)
                                show_image(maskedresult)



                            #dilationkernel = np.ones((2,2),np.uint8)
                            #dilatedimg = cv2.dilate(maskedresult, dilationkernel, iterations=1)

                            #show_image(maskedresult)
                            #show_image(dilatedimg)

                            dilatedimg = maskedresult

                            unknownvar,contours,h = cv2.findContours(cv2.cvtColor(dilatedimg,cv2.COLOR_BGR2GRAY),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)




                            if len(contours)>1:

                                #One of the contour pairs searched must include the point "far"
                                for cont in contours:
                                    if far in cont:
                                        far_cont = cont

                                contourpairs = []

                                for cont in contours:
                                    if not np.array_equal(cont,far_cont):
                                        contourpairs.append((far_cont,cont))

                                #contourpairs=itertools.combinations(contours, 2)
                                distances=[]
                                pointpairs=[]

                                for pair in contourpairs:
                                    for point1 in pair[0]:
                                        for point2 in pair[1]:
                                            distance=distance_formula(point1[0],point2[0])
                                            if distance>2:
                                                distances.append(distance)
                                                pointpairs.append((point1[0],point2[0]))



                                #extend connection lines slightly, unless line is vertical.

                                connectionpoint1 = tuple(pointpairs[distances.index(min(distances))][0])
                                connectionpoint2 = tuple(pointpairs[distances.index(min(distances))][1])
                                connectionlength = distance_formula(connectionpoint1,connectionpoint2)
                                

                                pixelstoextendby = 2

                                extconnectionpoint1 = list(connectionpoint1)
                                extconnectionpoint2 = list(connectionpoint2)



                                if connectionpoint1[0] > connectionpoint2[0]:
                                    extconnectionpoint1[0] = connectionpoint1[0] + pixelstoextendby
                                    extconnectionpoint2[0] = connectionpoint2[0] - pixelstoextendby

                                elif connectionpoint1[0] < connectionpoint2[0]:
                                    extconnectionpoint1[0] = connectionpoint1[0] - pixelstoextendby
                                    extconnectionpoint2[0] = connectionpoint2[0] + pixelstoextendby

                                #need to incorporate slope here, rather than just add same value to both x,y. Is this right?

                                if connectionpoint2[0] != connectionpoint1[0]:
                                    slope_between_conn_points = (connectionpoint2[1]-connectionpoint1[1])/float(connectionpoint2[0]-connectionpoint1[0])
                                    pixelstoextendby = pixelstoextendby * slope_between_conn_points



                                if connectionpoint1[1] > connectionpoint2[1]:
                                    extconnectionpoint1[1] = int(connectionpoint1[1] + pixelstoextendby)
                                    extconnectionpoint2[1] = int(connectionpoint2[1] - pixelstoextendby)

                                elif connectionpoint1[1] < connectionpoint2[1]:
                                    extconnectionpoint1[1] = int(connectionpoint1[1] - pixelstoextendby)
                                    extconnectionpoint2[1] = int(connectionpoint2[1] + pixelstoextendby)


                                extconnectionpoint1 = tuple(extconnectionpoint1)
                                extconnectionpoint2 = tuple(extconnectionpoint2)

                                cv2.line(updatingimg,extconnectionpoint1,extconnectionpoint2,(0,0,0),thickness=2)
                                cv2.line(breakupimg,extconnectionpoint1,extconnectionpoint2,(0,255,0),thickness=1)

                                if visualizemasking==True:
                                    cv2.circle(maskingvisualization,tuple(pointpairs[distances.index(min(distances))][0]),2,(255,0,255),-1)
                                    cv2.circle(maskingvisualization,tuple(pointpairs[distances.index(min(distances))][1]),2,(255,0,255),-1)
                                    show_image(maskingvisualization,imgname="maskingvisualization")
                                    show_image(maskedresult, imgname="maskedresult")




                            if breakuptesting==True:
                                show_image(breakupimg,imgname="breakupimg")
                                show_image(updatingimg,imgname="updatingimg")




    brokenupvertices=find_draw_contours(updatingimg,blocksize, minarea = avgarea/10, annotate=True,testing=testing,nofilter=True, contours_color = (255,255,255))


    indextoremove=list(set(indextoremove))

    for i in sorted(indextoremove,reverse=True):
        del clusterbreakupvertices[i]

    for i in brokenupvertices:
        clusterbreakupvertices.append(i[0])

    return clusterbreakupvertices


def discreteness_index_and_ellipse_fitting(edgecorrectedvertices,img,rows,cols,imgstdev):
    '''Calculate "discreteness index" for each particle, a measure of how dissimilar the inside of a detected border
    and the outside is. This is meant to be a measure of how succesful the detection was. High discreteness index 
    would mean the inside and outside are quite different which implies inside is particle while outside is background.

    This functions also fits ellipses to the detected shapes and calculates the new DI, if it is determined that the 
    ellipse fits the true shape of the particle better, is passed to the final list instead of the contour.

    :param list edgecorrectedvertices:list of detected particles.
    :param numpy.ndarray img: original image.
    :param int rows: number of rows in image.
    :param int cols: number of columns in image.
    :param float imgstdev: standard deviation of pixel intensities in image.

    :return list particlediscreteness: list of DI's of each particle.
    :return list ellipsefittedvertices: Corrected list of vertices of particles.

    TODO:
    - Format of list of vertices of fitted ellipses and original contours are not the same.
    This has lead to some ugly fixes, at the end of the function and in edge_correction.
    
    '''

    ellimg = img.copy()
    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    particlediscreteness=[]
    ellipsefittedvertices=[]
    convexitydefects=[]
    similarity_ellipse_vs_particle=[]
    dev_particle_vs_ellipse=[]

    particleindex=0

    #particle level.
    for i in edgecorrectedvertices:

        tempimg=img.copy()

        (xcom,ycom),contradius = cv2.minEnclosingCircle(i)
        area=cv2.contourArea(i)

        xcom=int(xcom)
        ycom=int(ycom)


        bufferfromborder=3
        hull=cv2.convexHull(i)

        if len(hull) > 4:

            ellipse=cv2.fitEllipse(np.array(hull))
            ellipsepoly=cv2.ellipse2Poly((int(ellipse[0][0]), 
                int(ellipse[0][1])),(int(ellipse[1][0]/2),int(ellipse[1][1]/2)),int(ellipse[2]), 0, 360, 5)      


            reds=[]
            blues=[]


            #vertex of particle level.
            for pt in i:
                if pt[0][0]>bufferfromborder and pt[0][0]<cols-bufferfromborder and pt[0][1]>bufferfromborder and pt[0][1]<rows-bufferfromborder:
                #Reds are points just inside the contour and blues are points just outside.

                    if pt[0][0]<xcom:
                        reds.append(img[pt[0][1]][pt[0][0]+bufferfromborder][0])
                        blues.append(img[pt[0][1]][pt[0][0]-bufferfromborder][0])

                    if pt[0][0]>xcom:
                        reds.append(img[pt[0][1]][pt[0][0]-bufferfromborder][0])
                        blues.append(img[pt[0][1]][pt[0][0]+bufferfromborder][0])

                    if pt[0][1]<ycom:
                        reds.append(img[pt[0][1]+bufferfromborder][pt[0][0]][0])
                        blues.append(img[pt[0][1]-bufferfromborder][pt[0][0]][0])

                    if pt[0][1]>ycom:
                        reds.append(img[pt[0][1]-bufferfromborder][pt[0][0]][0])
                        blues.append(img[pt[0][1]+bufferfromborder][pt[0][0]][0])

            ellipsereds=[]
            ellipseblues=[]

            for pt in list(ellipsepoly):

                pt=[pt]

                if pt[0][0]>bufferfromborder and pt[0][0]<cols-bufferfromborder and pt[0][1]>bufferfromborder and pt[0][1]<rows-bufferfromborder:
                #Reds are points just inside the contour and blues are points just outside.

                    if pt[0][0]<xcom:
                        ellipsereds.append(img[pt[0][1]][pt[0][0]+bufferfromborder][0])
                        ellipseblues.append(img[pt[0][1]][pt[0][0]-bufferfromborder][0])

                    if pt[0][0]>xcom:
                        ellipsereds.append(img[pt[0][1]][pt[0][0]-bufferfromborder][0])
                        ellipseblues.append(img[pt[0][1]][pt[0][0]+bufferfromborder][0])

                    if pt[0][1]<ycom:
                        ellipsereds.append(img[pt[0][1]+bufferfromborder][pt[0][0]][0])
                        ellipseblues.append(img[pt[0][1]-bufferfromborder][pt[0][0]][0])

                    if pt[0][1]>ycom:
                        ellipsereds.append(img[pt[0][1]-bufferfromborder][pt[0][0]][0])
                        ellipseblues.append(img[pt[0][1]+bufferfromborder][pt[0][0]][0])


            particleDI=float((sum(reds)-sum(blues))/len(i))
            ellipseDI=float((sum(ellipsereds)-sum(ellipseblues))/len(ellipsepoly))


            #Crude way of comparing colors of particle/ellipse.

            color_means = []
            color_stdevs = []

            #ellipsepoly and i 
            for particle in [ellipsepoly,i]:

                indivcolorimg=img.copy()

                cv2.fillPoly(indivcolorimg,[particle],(0,0,255))

                shapearea=[]

                (xcom,ycom),contradius = cv2.minEnclosingCircle(particle)
                xcom=int(xcom)
                ycom=int(ycom)
                contradius=int(contradius)

                #y and x's are inverted due to the way you scan a np. array..
                ystart=int(xcom-2*contradius) 
                yend=int(xcom+2*contradius)   
                xstart=int(ycom-2*contradius)
                xend=int(ycom+2*contradius)

                if ystart<0:
                    ystart=0
                if yend>cols:
                    yend=cols
                if xstart<0:
                    xstart=0
                if xend>rows:
                    xend=rows

                for x in range(xstart,xend):
                    for y in range(ystart,yend):
                        if indivcolorimg[x][y][2]==255 and indivcolorimg[x][y][0]==0:
                            shapearea.append(gimg[x][y])


                colormean=sum(shapearea)/len(shapearea)
                colorstdev=np.std(shapearea)

                color_means.append(colormean)
                color_stdevs.append(colorstdev)


            ellipse_color_mean, particle_color_mean = color_means
            ellipse_color_stdev, particle_color_stdev = color_stdevs

            if ellipseDI > particleDI and (ellipse_color_stdev < particle_color_stdev * 1.25):
                particlediscreteness.append(ellipseDI)
                ellipsefittedvertices.append(ellipsepoly)

            else:
                particlediscreteness.append(particleDI)
                ellipsefittedvertices.append(i)
                

            cv2.polylines(ellimg,[i],True,(0,255,0),thickness=1)
            cv2.polylines(ellimg,[ellipsepoly],True,(0,0,255),thickness=1)

            particleindex+=1





    #Hierarchy filtering.
    index_to_remove = []
    particles_to_remove = []
    particle_pairs = itertools.combinations(ellipsefittedvertices, 2)

    for pair in particle_pairs:
        first_in_second = 0.0
        second_in_first = 0.0

        for point0 in pair[0]:
            #ugly fix.
            if len(point0) < 2:
                point0 = tuple(point0[0])
            else:
                point0 = tuple(point0)

            #if point is inside the contour.
            if cv2.pointPolygonTest(pair[1],point0,False) == 1:
                first_in_second += 1

        for point1 in pair[1]:
            #ugly fix.
            if len(point1) < 2:
                point1 = tuple(point1[0])
            else:
                point1 = tuple(point1)

            if cv2.pointPolygonTest(pair[0],point1,False) == 1:
                second_in_first +=1


        #if more than half the vertices of the first cont are inside the second.
        if first_in_second / len(pair[0]) > 0.5:
            particles_to_remove.append(pair[0])

        if second_in_first / len(pair[1]) > 0.5:
            particles_to_remove.append(pair[1])


    for i in particles_to_remove:
        index_to_remove.append([np.array_equal(i,x) for x in ellipsefittedvertices].index(True))

    for i in sorted(index_to_remove,reverse=True):
        del ellipsefittedvertices[i]

    #show_image(ellimg)


    return particlediscreteness, ellipsefittedvertices



