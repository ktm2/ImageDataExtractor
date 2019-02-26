from PIL import Image
from pytesseract import image_to_string
from pytesseract import image_to_boxes

import cv2
import numpy as np
from contour_detections import *
from operator import itemgetter


def read_image(thresholdedimg, unit):
    '''Read text from an image using pytesseract. Note: pytesseract reads PIL format images, 
    so we convert our cv2 (numpy array) for this step.

    :param numpy.ndarray thresholdedimg: input image to be read.
    :param string unit: "nm" or "um" depending on what we're looking for.
    
    :return string greekengtextfromimage: text that was read.
     '''

    #convert input to appropriate format.
    imagetoread = Image.fromarray(thresholdedimg)

    #Only look for these characters.
    if unit == "um":
        cfg = 'pytess_config_um'
    else:
        cfg = "pytess_config_nm"

    #Read image using English alphabet.
    greekengtextfromimage= image_to_string(imagetoread,lang="eng",config=cfg)

    text_boxes = image_to_boxes(imagetoread, lang = "eng", config = cfg)

    return greekengtextfromimage, text_boxes


def threshold_for_text_reading(img,imgmean,imgstdev,threshold_to,thresholdvalue = None):
    '''Thresholds (binarize/filter) an image to facilitate either white or black text reading.
    :param numpy.ndarray img: image to be read
    :param int imgmean: mean pixel intensity in image
    :param float imgstdev: standard deviation of pixel intensities in image
    :param int threshold_to: 0 if looking for white text, 1 if looking for black text.
    :param int thresholdvalue: Optionally provide an exact threshold value [0,255]

    :return numpy.ndarray thresholdedimg: Filtered image.
    '''

    #Looking for white text.
    if threshold_to == 0:

        if thresholdvalue == None:
            thresholdvalue = imgmean+(3.5*imgstdev)
        
        thresholdmax = 235
        if thresholdvalue > thresholdmax:
            thresholdvalue = thresholdmax


        ret,thresholdedimg = cv2.threshold(img,thresholdvalue,255,threshold_to)


    #Look for black text.
    elif threshold_to == 1:

        if thresholdvalue == None:
            thresholdvalue = imgmean-(4*imgstdev)

        thresholdmin = 20
        if thresholdvalue < thresholdmin:
            thresholdvalue = thresholdmin

        ret,thresholdedimg = cv2.threshold(img,thresholdvalue,255,threshold_to)

    return thresholdedimg


def detect_scale_bar_and_inlays(gimg,imgmean,imgstdev,rows,cols,show=False):
    '''Visually detects scale bar of an image to determine the physical distance a pixel represents.
    :param numpy.ndarray gimg: input image
    :param int imgmean: mean pixel intensity in image
    :param float imgstdev: standard deviation of pixel intensities in image
    :param int rows: number of rows in image
    :param int cols: number of cols in image
    :param bool show: to monitor activities step by step

    :return tuple scalebar: (x,y,w,h) top left corner of scalebar, width, height.
    :return float scalevalue: number associated with scale bar.
    :return float conversion: unit of scalevalue 10e-6 for um, 10e-9 for nm.
    :return list inlaycoords: list of tuples in same format of obstructive inlays in image.

    TODO:
    - Inlay coords currently hashed out, will this be in?

    '''

    scalebar,scalevalue,conversion = None,None,None

    #Create new copy of image to work with.
    scalebarimg = gimg.copy()

    #Frame around edges.
    frame_constant = 3
    cv2.rectangle(scalebarimg,(0,0),(frame_constant,rows),(0,0,0),-1)
    cv2.rectangle(scalebarimg,(0,rows-frame_constant),(cols,rows),(0,0,0),-1)
    cv2.rectangle(scalebarimg,(0,0),(cols,frame_constant),(0,0,0),-1)
    cv2.rectangle(scalebarimg,(cols-frame_constant,0),(cols,rows),(0,0,0),-1)

    #Threshold image and look for white text.
    thresholdedimg=threshold_for_text_reading(scalebarimg,imgmean,imgstdev,0)
    scalebar,scalevalue,conversion,inlaycoords = find_text_and_bar(thresholdedimg,gimg,rows,cols,show=show,printer=show)

    #If nothing is found repeat process looking for black text.
    if scalebar == None and scalevalue == None and conversion == None:

        cv2.rectangle(scalebarimg,(0,0),(frame_constant,rows),(255,255,255),-1)
        cv2.rectangle(scalebarimg,(0,rows-frame_constant),(cols,rows),(255,255,255),-1)
        cv2.rectangle(scalebarimg,(0,0),(cols,frame_constant),(255,255,255),-1)
        cv2.rectangle(scalebarimg,(cols-frame_constant,0),(cols,rows),(255,255,255),-1)

        thresholdedimg_black=threshold_for_text_reading(scalebarimg,imgmean,imgstdev,1)
        scalebar,scalevalue,conversion,inlaycoords = find_text_and_bar(thresholdedimg_black,gimg,rows,cols,show=show,printer=show,black_text=True)



    return scalebar,scalevalue,conversion,inlaycoords


    #Find inlays or text blocks in image.

    # erosionkernel = np.ones((3,3),np.uint8)
    # dilationkernel = np.ones((8,10),np.uint8)

    # erodedimg = cv2.erode(thresholdedimg,erosionkernel,iterations=2)
    # dilatedimg = cv2.dilate(erodedimg, dilationkernel, iterations=5)


    # whiteinlays=[]
    # #White inlays.
    # filteredcontours,filteredvertices,boundingrectanglewidths,whiteinlays=find_draw_contours\
    # (dilatedimg,blocksize,minarea=100,maxarea=rows*cols/5,displayimg=False,bounding_rect=True,annotate=testing)

    # blackinlays=[]
    # #Black inlays.
    # if imgmean>70:
    #     filteredcontours,filteredvertices,boundingrectanglewidths,blackinlays=find_draw_contours\
    #     (threshold_for_text_reading(scalebarimg,imgmean,imgstdev,0),blocksize,minarea=100,maxarea=rows*cols/5,displayimg=False,bounding_rect=True,annotate=testing)


    # #Assert that inlays must be in edges of image.
    # inlaycoords=[inlay for inlay in whiteinlays if inlay[0]<0.1*cols or inlay[0]>0.9*cols or inlay[1]<0.1*rows or inlay[1]>0.9*rows]\
    # +[inlay for inlay in blackinlays if inlay[0]<0.1*cols or inlay[0]>0.9*cols or inlay[1]<0.1*rows or inlay[1]>0.9*rows]

    # if testing==True and len(inlaycoords)>0:

    #     for i in inlaycoords:
    #         cv2.rectangle(scalebarimg,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,255,255),thickness=2)

    #     show_image(scalebarimg)


    # return pixelscale,inlaycoords,scalebarboundingrectangle,scalevalue,conversion


def find_text_and_bar(thresholdedimg,gimg,rows,cols,show=False,printer=False,black_text=False):
        '''Locates scale information (bar, value and unit) in a thresholded image. (actual reading part of parent function).
        :param numpy.ndarray thresholdedimg: filtered input image, filtered for black or white text reading.
        :param numpy.ndarray gimg: original image in grayscale.
        :param int rows: number of rows in image.
        :param int cols: number of columns in image.
        :param bool show: display images step by step.
        :param bool printer: print what's going on step by step.
        :param bool black_text: True if we're looking for black text/bar, False if white.

        :return tuple scalebar: (x,y,w,h) top left corner of scalebar, width, height.
        :return float scalevalue: number associated with scale bar.
        :return float conversion: unit of scalevalue 10e-6 for um, 10e-9 for nm.
        :return list inlaycoords: list of tuples in same format of obstructive inlays in image.

        TODO:
        - Can this function be merged with parent function detect_scale_bar_and_inlays? Seems like
        everything is going on here.
        - This function can be made more efficient if we don't try to read every region of interest. A filtering
        step perhaps looking at the region_stdev can help filter out those that don't include text?

        '''

        scalebar = None
        conversion = None
        scalevalue = None
        boxes = None
        inlaycoords = []
        inlay = None
        
        #Make regions including text/info into large blobs so they can be easily identified.
        blurred_thresholdedimg=cv2.medianBlur(thresholdedimg,1)
        dilationkernel = np.ones((6,8),np.uint8)
        dilatedimg = cv2.dilate(blurred_thresholdedimg, dilationkernel, iterations=3)

        if black_text == False:
            restrict_rectangles_color = "white"
        else:
            restrict_rectangles_color = "black"


        filteredvertices,boundingrectanglewidths,boundingrectanglecoords=find_draw_contours \
        (dilatedimg,151,minarea=10,testing=show,annotate=False,bounding_rect=True,restrict_rectangles=False)


        potential_scale_bar_regions=[]
        area_index = 1

        for i in boundingrectanglecoords:

            if printer == True:
                print(area_index, " out of ", len(boundingrectanglecoords))

            is_box_in_box=False
            region_size = int(i[2])*int(i[3])

            if region_size>rows*cols/100:

                #First checking if the region we've identified is a box containing information. As opposed to text put directly on the image.

                box_filteredvertices,box_boundingrectanglewidths,box_boundingrectanglecoords=find_draw_contours\
                (thresholdedimg[i[1]:i[1]+i[3],i[0]:i[0]+i[2]],151,minarea=10,testing=show,annotate=show,bounding_rect=True,restrict_rectangles=True)
                
                box_boundingrectanglecoords.sort(key=itemgetter(2))
                box_boundingrectanglecoords.reverse()

                for potential_box in box_boundingrectanglecoords:

                    box_in_box=gimg[i[1]+potential_box[1]:i[1]+potential_box[1]+potential_box[3],i[0]+potential_box[0]:i[0]+potential_box[0]+potential_box[2]]

                    box_in_box_filteredvertices=find_draw_contours\
                (box_in_box,151,minarea=10,testing=show,annotate=show)


                    #Check that we've found box and not just the scale bar by mistake. By making sure shapes exist within this region.
                    if  len(box_in_box_filteredvertices) > 2:
                        i=(i[0]+potential_box[0],i[1]+potential_box[1],potential_box[2],potential_box[3])

                        is_box_in_box=True

                    #Only have to check widest box, so we can break after a single a loop.
                    break

                if printer == True:
                    print("is_box_in_box" , is_box_in_box)

                #These get set to None each loop unless both can be identified simultaneously further down.
                conversion = None
                boxes = None
                scalevalue = None

                #Cannot determine what exact binarization threshold works for each region, so we scan through 35,0 as thresholds in increments of 5 until
                #something matching our text criteria is found.

                for pixel_value in reversed(range(5,40,5)):

                    if conversion == None or scalevalue == None:

                        if black_text==False:

                            inverted=(255-gimg[i[1]:i[1]+i[3],i[0]:i[0]+i[2]])
                        else:
                            inverted=gimg[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
                        
                        h, w = inverted.shape[:2]
                        inverted = cv2.resize(inverted, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                        uv,region_surrounding_text_reading = cv2.threshold(inverted,pixel_value,255,0)
                        region_surrounding_text_reading = cv2.medianBlur(region_surrounding_text_reading,1)

                        greekengtextfromimage_edited_nm, edited_nm_boxes = read_image(region_surrounding_text_reading, unit = "nm")
                        greekengtextfromimage_edited_um, edited_um_boxes = read_image(region_surrounding_text_reading, unit = "um")

                        if printer == True:

                            print("greek&eng_edited_nm")
                            print(greekengtextfromimage_edited_nm)

                            print("greek&eng_edited_um")
                            print(greekengtextfromimage_edited_um)

                        if show == True:
                            show_image(region_surrounding_text_reading,0,"blurred")

                        #Check raw images only once, at the beginning.
                        if pixel_value == 35:
                            if show == True:
                                show_image(inverted,0,'raw')

                            greekengtextfromimage_raw_nm, raw_nm_boxes = read_image(inverted, unit = "nm")
                            greekengtextfromimage_raw_um, raw_um_boxes = read_image(inverted, unit = "um")

                            if printer == True:
                                print("greek&eng_raw_nm")
                                print(greekengtextfromimage_raw_nm)

                                print("greek&eng_raw_um")
                                print(greekengtextfromimage_raw_um)

                            if show == True:
                                show_image((255-inverted),0,'255-inverted')


                            greekengtextfromimage_raw_inverted_nm , inverted_nm_boxes = read_image((255-inverted), unit = "nm")
                            greekengtextfromimage_raw_inverted_um , inverted_um_boxes = read_image((255-inverted), unit = "um")

                            if printer == True:
                                print("greek&eng_raw_inverted_nm")
                                print(greekengtextfromimage_raw_inverted_nm)

                                print("greek&eng_raw_inverted_um")
                                print(greekengtextfromimage_raw_inverted_um)


                        #ORDER OF PRIORITY IS, THRESHOLDED/BLURRED, CORRECT INVERSION(ATTEMPTED), OTHER INVERSION.

                        #Criteria for determined text eligibility:
                        #First search for "um" (micron, since this gives less false positives) then "nm". Unit must be preceded by digit+space or digit to be considered.

                        digits='.0123456789 '

                        if 'um' in greekengtextfromimage_edited_um and any(char in digits for char in greekengtextfromimage_edited_um):
                            if len(greekengtextfromimage_edited_um.split("um")[0]) > 0:
                                if (greekengtextfromimage_edited_um.split("um")[0][-1] == " " or \
                                greekengtextfromimage_edited_um.split("um")[0][-1].isdigit() == True) :

                                    conversion=10**-6
                                    if all(char in digits for char in greekengtextfromimage_edited_um.split("um")[0]):
                                        scalevalue=greekengtextfromimage_edited_um.split("um")[0]
                                        boxes = edited_um_boxes


                        if scalevalue == None and "nm" in greekengtextfromimage_edited_nm and any(char in digits for char in greekengtextfromimage_edited_nm):
                            if len(greekengtextfromimage_edited_nm.split("nm")[0]) > 0:
                                if (greekengtextfromimage_edited_nm.split("nm")[0][-1] == " " or \
                                greekengtextfromimage_edited_nm.split("nm")[0][-1].isdigit() == True):

                                    conversion=10**-9
                                    if all(char in digits for char in greekengtextfromimage_edited_nm.split("nm")[0]):
                                        scalevalue=greekengtextfromimage_edited_nm.split("nm")[0]
                                        boxes = edited_nm_boxes

                        #Also check raw region only on the first pass just in case its legible.
                        elif pixel_value == 35:
                            if 'um' in greekengtextfromimage_raw_um and any(char in digits for char in greekengtextfromimage_raw_um): 
                                if len(greekengtextfromimage_raw_um.split("um")[0]) > 0:
                                    if ((greekengtextfromimage_raw_um.split("um")[0][-1] == " " and \
                                        greekengtextfromimage_raw_um.split("um")[0][-2].isdigit() == True) or \
                                    greekengtextfromimage_raw_um.split("um")[0][-1].isdigit() == True):

                                        conversion=10**-6

                                        if all(char in digits for char in greekengtextfromimage_raw_um.split("um")[0]):
                                            scalevalue=greekengtextfromimage_raw_um.split("um")[0]
                                            boxes = raw_um_boxes

                            if scalevalue == None and "nm" in greekengtextfromimage_raw_nm and any(char in digits for char in greekengtextfromimage_raw_nm): 
                                if len(greekengtextfromimage_raw_nm.split("nm")[0]) > 0:
                                    if (greekengtextfromimage_raw_nm.split("nm")[0][-1] == " " or \
                                    greekengtextfromimage_raw_nm.split("nm")[0][-1].isdigit() == True):

                                        conversion=10**-9

                                        if all(char in digits for char in greekengtextfromimage_raw_nm.split("nm")[0]):
                                            scalevalue=greekengtextfromimage_raw_nm.split("nm")[0]
                                            boxes = raw_nm_boxes

                            if scalevalue == None and 'um' in greekengtextfromimage_raw_inverted_um and any(char in digits for char in greekengtextfromimage_raw_inverted_um):  
                                if len(greekengtextfromimage_raw_inverted_um.split("um")[0]) > 0:
                                    if (greekengtextfromimage_raw_inverted_um.split("um")[0][-1] == " " or \
                                    greekengtextfromimage_raw_inverted_um.split("um")[0][-1].isdigit() == True):

                                        conversion=10**-6

                                        if all(char in digits for char in greekengtextfromimage_raw_inverted_um.split("um")[0]):
                                            scalevalue=greekengtextfromimage_raw_inverted_um.split("um")[0]
                                            boxes = inverted_um_boxes

                            if scalevalue == None and "nm" in greekengtextfromimage_raw_inverted_nm and any(char in digits for char in greekengtextfromimage_raw_inverted_nm): 
                                if len(greekengtextfromimage_raw_inverted_nm.split("nm")[0]) > 0:
                                    if (greekengtextfromimage_raw_inverted_nm.split("nm")[0][-1] == " " or \
                                    greekengtextfromimage_raw_inverted_nm.split("nm")[0][-1].isdigit() == True):

                                        conversion=10**-9

                                        if all(char in digits for char in greekengtextfromimage_raw_inverted_nm.split("nm")[0]):
                                            scalevalue=greekengtextfromimage_raw_inverted_nm.split("nm")[0]
                                            boxes = inverted_nm_boxes

                        #Filter out unrealistic scalevalues.
                        if scalevalue != None:
                            scalevalue=''.join(i for i in scalevalue if i.isdigit())
                            try:
                                if float(scalevalue) < 0.01 or float(scalevalue) > 1000:
                                    scalevalue = None
                            except ValueError:
                                scalevalue = None

                    #If both a scalevalue and a conversion that meet all criteria are found we can break out of the incremental thresholding of the region.
                    else:
                        break


                if conversion!=None and scalevalue!=None:
                    scalevalue = float(''.join(i for i in scalevalue if i.isdigit()))
                    potential_scale_bar_regions.append([i,conversion,scalevalue,is_box_in_box,boxes])
                    if printer == True:

                        print("worked at pixel_value ", str(pixel_value+5)) #since loop is stopped one loop too late
                        print("found eligible text, appended info")
                        print([i,conversion,scalevalue])
                else:
                    boxes = None


            area_index+=1

        #Now that we have regions that include a number + unit, we need to search their greater regions for scale bars.
        #Since the bar will be somewhere near this text.

        potential_scale_bars=[]
        for potential_region in potential_scale_bar_regions:
            region = potential_region[0]



            #If the region is a "box_in_box" we don't have to expand the search area as the bar will be within the same box.
            if potential_region[3] == True:
                region_border_buffer = 0

                regionsurroundingscalebar=gimg[region[1]:region[1]+region[3]+region_border_buffer,\
                region[0]:region[0]+region[2]+region_border_buffer]
                surrounding_region_y1 = region[0]
                surrounding_region_x1 = region[1]



            #If not box_in_box we expand our search region.
            else:
                surrounding_region_x1=region[1]-region[3]

                if surrounding_region_x1<0:
                    surrounding_region_x1=0

                surrounding_region_x2=region[1]+2*region[3]

                if surrounding_region_x2>rows:
                    surrounding_region_x2=rows

                surrounding_region_y1=region[0]-region[2]

                if surrounding_region_y1<0:
                    surrounding_region_y1=0

                surrounding_region_y2=region[0]+2*region[2]

                if surrounding_region_y2>cols:
                    surrounding_region_y2=cols


                regionsurroundingscalebar=thresholdedimg[surrounding_region_x1:surrounding_region_x2, \
                surrounding_region_y1:surrounding_region_y2]

                restrict_rectangles_color = None

            #Search the area for rectangles.
            filteredvertices,boundingrectanglewidths,boundingrectanglecoords=find_draw_contours\
        (regionsurroundingscalebar,151,minarea=10,nofilter=True,testing=show,annotate=show,bounding_rect=True,restrict_rectangles=True,restrict_rectangles_color=restrict_rectangles_color)


            #Quick filter on detected rectangle, w > h by at least 1.3 and width smaller than img width/1.5
            first_scale_bar_rectangles = [a for a in boundingrectanglecoords if float(a[2])/a[3] >= 1.3 and a[2] < cols/1.5]

            potential_scale_bar_rectangles = []

            #Checks of whether  scale bars of solid color. (Not a chunk of the image by mistake)
            
            for i in first_scale_bar_rectangles:

                fsb = gimg[i[1]+surrounding_region_x1:i[1]+surrounding_region_x1+i[3],i[0]+surrounding_region_y1:i[0]+surrounding_region_y1+i[2]].flatten()

                fsb_stdev = np.std(fsb)
                fsb_mean = sum(fsb)/len(fsb)

                #uniform color that's either black or white.
                if fsb_stdev < 50 and (fsb_mean>200 or fsb_mean < 30):
                    potential_scale_bar_rectangles.append(i)



            scalebarboundingrectangle=None
            scalebarwidth=None

            #If we still have multiple candidates in this region, we just go with the widest one.
            potential_scale_bar_widths=[a[2] for a in potential_scale_bar_rectangles]

            if len(potential_scale_bar_widths) > 0:
                scalebarwidth=max(potential_scale_bar_widths)
                scalebarboundingrectangle=potential_scale_bar_rectangles[potential_scale_bar_widths.index(scalebarwidth)]

                boxes_relative = potential_region[-1]
                boxes = []
                
                for b in boxes_relative.splitlines():
                    b = b.split(' ')

                    if str(b[0]) in "0123456789num":

                        b = [int(i) for i in b[1:5]]

                        boxes.append([(b[0]+region[0],region[3]-b[1]+region[1]),(b[2]+region[0],region[3]-b[3]+region[1])])



                potential_scale_bars.append([(int(scalebarboundingrectangle[0]+surrounding_region_y1),int(scalebarboundingrectangle[1]+surrounding_region_x1)\
                ,int(scalebarboundingrectangle[2]),int(scalebarboundingrectangle[3])),potential_region[1],potential_region[2],boxes])

        #If we have multiple candidates, in multiple regions of interest, we go with the widest one globally.
        widths=[a[0][2] for a in potential_scale_bars]

        if len(widths) > 0:
            scalebarwidth=max(widths)
            scalebar,conversion,scalevalue,boxes = potential_scale_bars[widths.index(max(widths))]

        if all(i is not None for i in [scalebar, conversion, scalevalue, boxes]):


            ##more sophisticated inlay construction, going to pass image_to_boxes during reading and use it here.
            # using bounding boxes of detected characters + scalebar to determine accurate inlay location.

            box_xs = []
            box_ys = []

            for box in boxes:
                box_xs.append(box[0][0])
                box_xs.append(box[1][0])

                box_ys.append(box[0][1])
                box_ys.append(box[1][1])

            #     cv2.rectangle(gimg,(box[0][0],box[0][1]),(box[1][0],box[1][1]),
            # (255,255,255),thickness=1)

            # cv2.rectangle(gimg,(scalebar[0],scalebar[1]),(scalebar[0]+scalebar[2],scalebar[1]+scalebar[3]),
            # (255,255,255),thickness=1)

            box_xs.append(scalebar[0])
            box_xs.append(scalebar[0]+scalebar[2])

            box_ys.append(scalebar[1])
            box_ys.append(scalebar[1]+scalebar[3])


            inlaycoords.append((min(box_xs),min(box_ys),max(box_xs)-min(box_xs),max(box_ys)-min(box_ys)))

        #Inlaycoords only has scalebar region now, other detected ones will be appended from here.

        return scalebar,scalevalue,conversion,inlaycoords







