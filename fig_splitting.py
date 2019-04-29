import numpy as np
import cv2
from img_utils import *



def line_detection_and_split(gimg, show = False, eval_img = True):
    '''Detects vertical and horizontal lines in a figure along a (2,3,4)x(2,3,4) grid and splits
    into constituent images.

    :param numpy.ndarray gimg: grayscale input figure.
    :param bool show: display steps for debugging.
    :param bool eval_img: output an additional image for debugging.

    :return list img_split_vertically: list of images in figure
    :return numpy.ndarray evaluation_img: evaluation image, none if eval_img = False

    TODO:
    - checking along grid manually is inefficient, can do this recursively:
    - this can be adapted to unevenly split figures if the detected straight lines can be
    filtered more intelligently.
    - getting rid of debugging related objects.

    '''

    #Measure input figure.
    rows = len(gimg)
    cols = len(gimg[0])

    #Create copies of figure to draw on (for debugging/evaluation)
    drawingimg = cv2.cvtColor(gimg,cv2.COLOR_GRAY2RGB)
    evaluation_img = drawingimg.copy()

    #Apply visual filtereing and edge detection.
    low_threshold = 200
    high_threshold = 255
    edges = cv2.Canny(gimg, low_threshold, high_threshold)

    if show == True:
        show_image(edges)
        print cols,rows


    # Hough Lines Detection, from stackoverflow.
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = int(0.5*min(rows,cols))  # minimum number of pixels making up a line
    max_line_gap = int(0.5*min(rows,cols))  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments.
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

    if lines is None:
        print "no straight lines found"
        return None, evaluation_img



    horizontal_lines = []
    vertical_lines = []

    # X describes column number, Y describes row number.
    # take only horizonta;/vertical lines.
    for line in lines:
        for x1,y1,x2,y2 in line:

            if x1 == x2:
                vertical_lines.append(line[0])
                cv2.line(drawingimg,(x1,y1),(x2,y2),(255,0,0),1)

            if y1 == y2:
                horizontal_lines.append(line[0])
                cv2.line(drawingimg,(x1,y1),(x2,y2),(255,0,0),1)




    x1s=[line[0] for line in vertical_lines]
    y1s=[line[1] for line in horizontal_lines]

    #Checking along which fractions of image dimensions straight lines exist.
    horizontally_in_half = 0
    vertically_in_half = 0
    horizontally_in_thirds = 0
    vertically_in_thirds = 0
    horizontally_in_four = 0
    vertically_in_four = 0

    #Fractional tolerance along splitting points.
    tolerance = 0.015

    for line in horizontal_lines:

        if (rows/2) - int(rows*tolerance) < line[1] == line[3] < (rows/2) + int(rows*tolerance):
            horizontally_in_half += 1
            horizontally_in_four += 1

        if (rows/3) - int(rows*tolerance) < line[1] == line[3] < (rows/3) + int(rows*tolerance):
            horizontally_in_thirds += 1

        if (2*rows/3) - int(rows*tolerance) < line[1] == line[3] < (2*rows/3) + int(rows*tolerance):
            horizontally_in_thirds += 1

        if (rows/4) - int(rows*tolerance) < line[1] == line[3] < (rows/4) + int(rows*tolerance):
            horizontally_in_four += 1

        if (3*rows/4) - int(rows*tolerance) < line[1] == line[3] < (3*rows/4) + int(rows*tolerance):
            horizontally_in_four += 1

    for line in vertical_lines:

        if (cols/2) - int(cols*tolerance) < line[0] == line[2] < (cols/2) + int(cols*tolerance):
            vertically_in_half += 1
            vertically_in_four += 1

        if (cols/3) - int(cols*tolerance) < line[0] == line[2] < (cols/3) + int(cols*tolerance):
            vertically_in_thirds += 1

        if (2*cols/3) - int(cols*tolerance) < line[0] == line[2] < (2*cols/3) + int(cols*tolerance):
            vertically_in_thirds += 1
        
        if (cols/4) - int(cols*tolerance) < line[0] == line[2] < (cols/4) + int(cols*tolerance):
            vertically_in_four += 1

        if (3*cols/4) - int(cols*tolerance) < line[0] == line[2] < (3*cols/4) + int(cols*tolerance):
            vertically_in_four += 1


    if show == True:
        print "horizontally_in_half", horizontally_in_half
        print "horizontally_in_thirds", horizontally_in_thirds
        print "horizontally_in_four", horizontally_in_four
        print "vertically_in_half", vertically_in_half
        print "vertically_in_thirds", vertically_in_thirds
        print "vertically_in_four", vertically_in_four

        show_image(drawingimg)



    img_split_horizontally = []

    #Split horizontally in four, using the previously detected straight lines closest to the "grid" lines.

    if horizontally_in_four > horizontally_in_half > 0:

        #Finding where to split.
        distance_to_1st_splitline=[line[1] - rows/4 for line in horizontal_lines]

        from_bottom_1st = [a for a in distance_to_1st_splitline if rows * tolerance >= a >= 0]
        from_top_1st = [a for a in distance_to_1st_splitline if 0 >= a >= -rows * tolerance]


        if len(from_bottom_1st) == 0:
            closest_from_bottom_1st = rows/4
        else:
            closest_from_bottom_1st=y1s[distance_to_1st_splitline.index(min(from_bottom_1st))]

        if len(from_top_1st) == 0:
            closest_from_top_1st = rows/4
        else:
            closest_from_top_1st=y1s[distance_to_1st_splitline.index(max(from_top_1st))]


        distance_to_2nd_splitline=[line[1] - 2*rows/4 for line in horizontal_lines]

        from_bottom_2nd = [a for a in distance_to_2nd_splitline if rows * tolerance >= a >= 0]
        from_top_2nd = [a for a in distance_to_2nd_splitline if 0 >= a >= -rows * tolerance]


        if len(from_bottom_2nd) == 0:
            closest_from_bottom_2nd = 2*rows/4
        else:
            closest_from_bottom_2nd=y1s[distance_to_2nd_splitline.index(min(from_bottom_2nd))]

        if len(from_top_2nd) == 0:
            closest_from_top_2nd = 2*rows/4
        else:
            closest_from_top_2nd=y1s[distance_to_2nd_splitline.index(max(from_top_2nd))]



        distance_to_3rd_splitline=[line[1] - 3*rows/4 for line in horizontal_lines]

        from_bottom_3rd = [a for a in distance_to_3rd_splitline if rows * tolerance >= a >= 0]
        from_top_3rd = [a for a in distance_to_3rd_splitline if 0 >= a >= -rows * tolerance]


        if len(from_bottom_3rd) == 0:
            closest_from_bottom_3rd = 3*rows/4
        else:
            closest_from_bottom_3rd=y1s[distance_to_3rd_splitline.index(min(from_bottom_3rd))]

        if len(from_top_3rd) == 0:
            closest_from_top_3rd = 3*rows/4
        else:
            closest_from_top_3rd=y1s[distance_to_3rd_splitline.index(max(from_top_3rd))]


        #Splitting the image and appending to output.
        img_0 = gimg[0:closest_from_top_1st,0:cols]
        img_1 = gimg[closest_from_bottom_1st:closest_from_top_2nd,0:cols]
        img_2 = gimg[closest_from_bottom_2nd:closest_from_top_3rd,0:cols]
        img_3 = gimg[closest_from_bottom_3rd:rows,0:cols]

        img_split_horizontally.append(img_0)
        img_split_horizontally.append(img_1)
        img_split_horizontally.append(img_2)
        img_split_horizontally.append(img_3)


        if eval_img == True:
            cv2.line(evaluation_img,(0,closest_from_top_1st),(cols,closest_from_top_1st),(0,0,255),1)
            cv2.line(evaluation_img,(0,closest_from_bottom_1st),(cols,closest_from_bottom_1st),(0,0,255),1)

            cv2.line(evaluation_img,(0,closest_from_top_2nd),(cols,closest_from_top_2nd),(0,0,255),1)
            cv2.line(evaluation_img,(0,closest_from_bottom_2nd),(cols,closest_from_bottom_2nd),(0,0,255),1)

            cv2.line(evaluation_img,(0,closest_from_top_3rd),(cols,closest_from_top_3rd),(0,0,255),1)
            cv2.line(evaluation_img,(0,closest_from_bottom_3rd),(cols,closest_from_bottom_3rd),(0,0,255),1)


    elif horizontally_in_half > 0:

        distance_to_midline=[line[1] - rows/2 for line in horizontal_lines]

        from_bottom = [a for a in distance_to_midline if rows * tolerance >= a >= 0]
        from_top = [a for a in distance_to_midline if 0 >= a >= -rows * tolerance]

        if len(from_bottom) == 0:
            closest_from_bottom = rows/2
        else:
            closest_from_bottom=y1s[distance_to_midline.index(min(from_bottom))]

        if len(from_top) == 0:
            closest_from_top = rows/2
        else:
            closest_from_top=y1s[distance_to_midline.index(max(from_top))]


        img_top_half = gimg[0:closest_from_top,0:cols]
        img_bottom_half = gimg[closest_from_bottom:rows,0:cols]

        img_split_horizontally.append(img_top_half)
        img_split_horizontally.append(img_bottom_half)

        if eval_img == True:
            cv2.line(evaluation_img,(0,closest_from_top),(cols,closest_from_top),(0,0,255),1)
            cv2.line(evaluation_img,(0,closest_from_bottom),(cols,closest_from_bottom),(0,0,255),1)



    elif horizontally_in_thirds > 1:

        distance_to_1st_splitline=[line[1] - rows/3 for line in horizontal_lines]

        from_bottom_1st = [a for a in distance_to_1st_splitline if rows * tolerance >= a >= 0]
        from_top_1st = [a for a in distance_to_1st_splitline if 0 >= a >= -rows * tolerance]


        if len(from_bottom_1st) == 0:
            closest_from_bottom_1st = rows/3
        else:
            closest_from_bottom_1st=y1s[distance_to_1st_splitline.index(min(from_bottom_1st))]

        if len(from_top_1st) == 0:
            closest_from_top_1st = rows/3
        else:
            closest_from_top_1st=y1s[distance_to_1st_splitline.index(max(from_top_1st))]


        distance_to_2nd_splitline=[line[1] - 2*rows/3 for line in horizontal_lines]

        from_bottom_2nd = [a for a in distance_to_2nd_splitline if rows * tolerance >= a >= 0]
        from_top_2nd = [a for a in distance_to_2nd_splitline if 0 >= a >= -rows * tolerance]


        if len(from_bottom_2nd) == 0:
            closest_from_bottom_2nd = 2*rows/3
        else:
            closest_from_bottom_2nd=y1s[distance_to_2nd_splitline.index(min(from_bottom_2nd))]

        if len(from_top_2nd) == 0:
            closest_from_top_2nd = 2*rows/3
        else:
            closest_from_top_2nd=y1s[distance_to_2nd_splitline.index(max(from_top_2nd))]


        img_0 = gimg[0:closest_from_top_1st,0:cols]
        img_1 = gimg[closest_from_bottom_1st:closest_from_top_2nd,0:cols]
        img_2 = gimg[closest_from_bottom_2nd:rows,0:cols]

        img_split_horizontally.append(img_0)
        img_split_horizontally.append(img_1)
        img_split_horizontally.append(img_2)

        if eval_img == True:
            cv2.line(evaluation_img,(0,closest_from_top_1st),(cols,closest_from_top_1st),(0,0,255),1)
            cv2.line(evaluation_img,(0,closest_from_bottom_1st),(cols,closest_from_bottom_1st),(0,0,255),1)

            cv2.line(evaluation_img,(0,closest_from_top_2nd),(cols,closest_from_top_2nd),(0,0,255),1)
            cv2.line(evaluation_img,(0,closest_from_bottom_2nd),(cols,closest_from_bottom_2nd),(0,0,255),1)

    else:
        img_split_horizontally.append(gimg)


    if show == True:

        for i in img_split_horizontally:
            show_image(i)


    #Same process as above but vertical splits.
    img_split_vertically = []

    if vertically_in_four > vertically_in_half > 0:

        distance_to_1st_splitline=[line[0] - cols/4 for line in vertical_lines]

        from_right_1st = [a for a in distance_to_1st_splitline if cols * tolerance >= a >= 0]
        from_left_1st = [a for a in distance_to_1st_splitline if 0 >= a >= -cols * tolerance]


        if len(from_right_1st) == 0:
            closest_from_right_1st = cols/4
        else:
            closest_from_right_1st=x1s[distance_to_1st_splitline.index(min(from_right_1st))]

        if len(from_left_1st) == 0:
            closest_from_left_1st = cols/4
        else:
            closest_from_left_1st=x1s[distance_to_1st_splitline.index(max(from_left_1st))]


        distance_to_2nd_splitline=[line[0] - 2*cols/4 for line in vertical_lines]

        from_right_2nd = [a for a in distance_to_2nd_splitline if cols * tolerance >= a >= 0]
        from_left_2nd = [a for a in distance_to_2nd_splitline if 0 >= a >= -cols * tolerance]


        if len(from_right_2nd) == 0:
            closest_from_right_2nd = 2*cols/4
        else:
            closest_from_right_2nd=x1s[distance_to_2nd_splitline.index(min(from_right_2nd))]

        if len(from_left_2nd) == 0:
            closest_from_left_2nd = 2*cols/4
        else:
            closest_from_left_2nd=x1s[distance_to_2nd_splitline.index(max(from_left_2nd))]


        distance_to_3rd_splitline=[line[0] - 3*cols/4 for line in vertical_lines]

        from_right_3rd = [a for a in distance_to_3rd_splitline if cols * tolerance >= a >= 0]
        from_left_3rd = [a for a in distance_to_3rd_splitline if 0 >= a >= -cols * tolerance]


        if len(from_right_3rd) == 0:
            closest_from_right_3rd = 3*cols/4
        else:
            closest_from_right_3rd=x1s[distance_to_3rd_splitline.index(min(from_right_3rd))]

        if len(from_left_3rd) == 0:
            closest_from_left_3rd = 3*cols/4
        else:
            closest_from_left_3rd=x1s[distance_to_3rd_splitline.index(max(from_left_3rd))]



        if len(img_split_horizontally) == 1:

            img_0 = img_split_horizontally[0]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:closest_from_left_3rd]
            img_0_3 = img_0[0:len(img_0),closest_from_right_3rd:len(img_0[0])]

            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)
            img_split_vertically.append(img_0_3)


        if len(img_split_horizontally) == 2:


            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:closest_from_left_3rd]
            img_0_3 = img_0[0:len(img_0),closest_from_right_3rd:len(img_0[0])]


            img_1_0 = img_1[0:len(img_1),0:closest_from_left_1st]
            img_1_1 = img_1[0:len(img_1),closest_from_right_1st:closest_from_left_2nd]
            img_1_2 = img_1[0:len(img_1),closest_from_right_2nd:closest_from_left_3rd]
            img_1_3 = img_1[0:len(img_1),closest_from_right_3rd:len(img_1[0])]


            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)
            img_split_vertically.append(img_0_3)

            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_1_2)
            img_split_vertically.append(img_1_3)



        if len(img_split_horizontally) == 3:

            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]
            img_2 = img_split_horizontally[2]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:closest_from_left_3rd]
            img_0_3 = img_0[0:len(img_0),closest_from_right_3rd:len(img_0[0])]

            img_1_0 = img_1[0:len(img_1),0:closest_from_left_1st]
            img_1_1 = img_1[0:len(img_1),closest_from_right_1st:closest_from_left_2nd]
            img_1_2 = img_1[0:len(img_1),closest_from_right_2nd:closest_from_left_3rd]
            img_1_3 = img_1[0:len(img_1),closest_from_right_3rd:len(img_1[0])]

            img_2_0 = img_2[0:len(img_2),0:closest_from_left_1st]
            img_2_1 = img_2[0:len(img_2),closest_from_right_1st:closest_from_left_2nd]
            img_2_2 = img_2[0:len(img_2),closest_from_right_2nd:closest_from_left_3rd]
            img_2_3 = img_2[0:len(img_2),closest_from_right_3rd:len(img_2[0])]



            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)
            img_split_vertically.append(img_0_3)


            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_1_2)
            img_split_vertically.append(img_1_3)

            img_split_vertically.append(img_2_0)
            img_split_vertically.append(img_2_1)
            img_split_vertically.append(img_2_2)
            img_split_vertically.append(img_2_3)


        if len(img_split_horizontally) == 4:

            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]
            img_2 = img_split_horizontally[2]
            img_3 = img_split_horizontally[3]


            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:closest_from_left_3rd]
            img_0_3 = img_0[0:len(img_0),closest_from_right_3rd:len(img_0[0])]


            img_1_0 = img_1[0:len(img_1),0:closest_from_left_1st]
            img_1_1 = img_1[0:len(img_1),closest_from_right_1st:closest_from_left_2nd]
            img_1_2 = img_1[0:len(img_1),closest_from_right_2nd:closest_from_left_3rd]
            img_1_3 = img_1[0:len(img_1),closest_from_right_3rd:len(img_1[0])]

            img_2_0 = img_2[0:len(img_2),0:closest_from_left_1st]
            img_2_1 = img_2[0:len(img_2),closest_from_right_1st:closest_from_left_2nd]
            img_2_2 = img_2[0:len(img_2),closest_from_right_2nd:closest_from_left_3rd]
            img_2_3 = img_2[0:len(img_2),closest_from_right_3rd:len(img_2[0])]

            img_3_0 = img_3[0:len(img_3),0:closest_from_left_1st]
            img_3_1 = img_3[0:len(img_3),closest_from_right_1st:closest_from_left_2nd]
            img_3_2 = img_3[0:len(img_3),closest_from_right_2nd:closest_from_left_3rd]
            img_3_3 = img_3[0:len(img_3),closest_from_right_3rd:len(img_3[0])]


            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)
            img_split_vertically.append(img_0_3)

            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_1_2)
            img_split_vertically.append(img_1_3)

            img_split_vertically.append(img_2_0)
            img_split_vertically.append(img_2_1)
            img_split_vertically.append(img_2_2)
            img_split_vertically.append(img_2_3)

            img_split_vertically.append(img_3_0)
            img_split_vertically.append(img_3_1)
            img_split_vertically.append(img_3_2)
            img_split_vertically.append(img_3_3)


        if eval_img == True:
            cv2.line(evaluation_img,(closest_from_left_1st,0),(closest_from_left_1st,rows),(0,0,255),1)
            cv2.line(evaluation_img,(closest_from_right_1st,0),(closest_from_right_1st,rows),(0,0,255),1)

            cv2.line(evaluation_img,(closest_from_left_2nd,0),(closest_from_left_2nd,rows),(0,0,255),1)
            cv2.line(evaluation_img,(closest_from_right_2nd,0),(closest_from_right_2nd,rows),(0,0,255),1)

            cv2.line(evaluation_img,(closest_from_left_3rd,0),(closest_from_left_3rd,rows),(0,0,255),1)
            cv2.line(evaluation_img,(closest_from_right_3rd,0),(closest_from_right_3rd,rows),(0,0,255),1)






    elif vertically_in_half > 0:

        distance_to_midline=[line[0] - cols/2 for line in vertical_lines]

        from_left = [a for a in distance_to_midline if cols * tolerance >= a >= 0]
        from_right = [a for a in distance_to_midline if 0 >= a >= -cols * tolerance]

        if len(from_left) == 0:
            closest_from_left = cols/2
        else:
            closest_from_left=x1s[distance_to_midline.index(min(from_left))]

        if len(from_right) == 0:
            closest_from_right = cols/2
        else:
            closest_from_right=x1s[distance_to_midline.index(max(from_right))]


        if len(img_split_horizontally) == 1:

            img_0 = img_split_horizontally[0]


            img_0_0 = img_0[0:len(img_0),0:closest_from_left]
            img_0_1 = img_0[0:len(img_0),closest_from_right:cols]

            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)


        if len(img_split_horizontally) == 2:

            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left]
            img_0_1 = img_0[0:len(img_0),closest_from_right:cols]
            img_1_0 = img_1[0:len(img_1),0:closest_from_left]
            img_1_1 = img_1[0:len(img_1),closest_from_right:cols]


            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)


        if len(img_split_horizontally) == 3:

            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]
            img_2 = img_split_horizontally[2]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left]
            img_0_1 = img_0[0:len(img_0),closest_from_right:cols]
            img_1_0 = img_1[0:len(img_1),0:closest_from_left]
            img_1_1 = img_1[0:len(img_1),closest_from_right:cols]
            img_2_0 = img_2[0:len(img_2),0:closest_from_left]
            img_2_1 = img_2[0:len(img_2),closest_from_right:cols]

            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_2_0)
            img_split_vertically.append(img_2_1)

        if len(img_split_horizontally) == 4:

            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]
            img_2 = img_split_horizontally[2]
            img_3 = img_split_horizontally[3]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left]
            img_0_1 = img_0[0:len(img_0),closest_from_right:cols]
            img_1_0 = img_1[0:len(img_1),0:closest_from_left]
            img_1_1 = img_1[0:len(img_1),closest_from_right:cols]
            img_2_0 = img_2[0:len(img_2),0:closest_from_left]
            img_2_1 = img_2[0:len(img_2),closest_from_right:cols]
            img_3_0 = img_2[0:len(img_3),0:closest_from_left]
            img_3_1 = img_2[0:len(img_3),closest_from_right:cols]

            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_2_0)
            img_split_vertically.append(img_2_1)
            img_split_vertically.append(img_3_0)
            img_split_vertically.append(img_3_1)



        if eval_img == True:
            cv2.line(evaluation_img,(closest_from_left,0),(closest_from_left,rows),(0,0,255),1)
            cv2.line(evaluation_img,(closest_from_right,0),(closest_from_right,rows),(0,0,255),1)

    elif vertically_in_thirds > 1:

        distance_to_1st_splitline=[line[0] - cols/3 for line in vertical_lines]

        from_right_1st = [a for a in distance_to_1st_splitline if cols * tolerance >= a >= 0]
        from_left_1st = [a for a in distance_to_1st_splitline if 0 >= a >= -cols * tolerance]


        if len(from_right_1st) == 0:
            closest_from_right_1st = cols/3
        else:
            closest_from_right_1st=x1s[distance_to_1st_splitline.index(min(from_right_1st))]

        if len(from_left_1st) == 0:
            closest_from_left_1st = cols/3
        else:
            closest_from_left_1st=x1s[distance_to_1st_splitline.index(max(from_left_1st))]


        distance_to_2nd_splitline=[line[0] - 2*cols/3 for line in vertical_lines]

        from_right_2nd = [a for a in distance_to_2nd_splitline if cols * tolerance >= a >= 0]
        from_left_2nd = [a for a in distance_to_2nd_splitline if 0 >= a >= -cols * tolerance]


        if len(from_right_2nd) == 0:
            closest_from_right_2nd = 2*cols/3
        else:
            closest_from_right_2nd=x1s[distance_to_2nd_splitline.index(min(from_right_2nd))]

        if len(from_left_2nd) == 0:
            closest_from_left_2nd = 2*cols/3
        else:
            closest_from_left_2nd=x1s[distance_to_2nd_splitline.index(max(from_left_2nd))]


        if len(img_split_horizontally) == 1:

            img_0 = img_split_horizontally[0]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:len(img_0[0])]

            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)


        if len(img_split_horizontally) == 2:


            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:len(img_0[0])]


            img_1_0 = img_1[0:len(img_1),0:closest_from_left_1st]
            img_1_1 = img_1[0:len(img_1),closest_from_right_1st:closest_from_left_2nd]
            img_1_2 = img_1[0:len(img_1),closest_from_right_2nd:len(img_1[0])]



            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)
            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_1_2)



        if len(img_split_horizontally) == 3:

            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]
            img_2 = img_split_horizontally[2]

            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:len(img_0[0])]


            img_1_0 = img_1[0:len(img_1),0:closest_from_left_1st]
            img_1_1 = img_1[0:len(img_1),closest_from_right_1st:closest_from_left_2nd]
            img_1_2 = img_1[0:len(img_1),closest_from_right_2nd:len(img_1[0])]

            img_2_0 = img_1[0:len(img_2),0:closest_from_left_1st]
            img_2_1 = img_1[0:len(img_2),closest_from_right_1st:closest_from_left_2nd]
            img_2_2 = img_1[0:len(img_2),closest_from_right_2nd:len(img_2[0])]



            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)
            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_1_2)
            img_split_vertically.append(img_2_0)
            img_split_vertically.append(img_2_1)
            img_split_vertically.append(img_2_2)


        if len(img_split_horizontally) == 4:

            img_0 = img_split_horizontally[0]
            img_1 = img_split_horizontally[1]
            img_2 = img_split_horizontally[2]
            img_3 = img_split_horizontally[3]


            img_0_0 = img_0[0:len(img_0),0:closest_from_left_1st]
            img_0_1 = img_0[0:len(img_0),closest_from_right_1st:closest_from_left_2nd]
            img_0_2 = img_0[0:len(img_0),closest_from_right_2nd:len(img_0[0])]


            img_1_0 = img_1[0:len(img_1),0:closest_from_left_1st]
            img_1_1 = img_1[0:len(img_1),closest_from_right_1st:closest_from_left_2nd]
            img_1_2 = img_1[0:len(img_1),closest_from_right_2nd:len(img_1[0])]

            img_2_0 = img_1[0:len(img_2),0:closest_from_left_1st]
            img_2_1 = img_1[0:len(img_2),closest_from_right_1st:closest_from_left_2nd]
            img_2_2 = img_1[0:len(img_2),closest_from_right_2nd:len(img_2[0])]

            img_3_0 = img_1[0:len(img_3),0:closest_from_left_1st]
            img_3_1 = img_1[0:len(img_3),closest_from_right_1st:closest_from_left_2nd]
            img_3_2 = img_1[0:len(img_3),closest_from_right_2nd:len(img_2[0])]



            img_split_vertically.append(img_0_0)
            img_split_vertically.append(img_0_1)
            img_split_vertically.append(img_0_2)

            img_split_vertically.append(img_1_0)
            img_split_vertically.append(img_1_1)
            img_split_vertically.append(img_1_2)

            img_split_vertically.append(img_2_0)
            img_split_vertically.append(img_2_1)
            img_split_vertically.append(img_2_2)

            img_split_vertically.append(img_3_0)
            img_split_vertically.append(img_3_1)
            img_split_vertically.append(img_3_2)

        if eval_img == True:
            cv2.line(evaluation_img,(closest_from_left_1st,0),(closest_from_left_1st,rows),(0,0,255),1)
            cv2.line(evaluation_img,(closest_from_right_1st,0),(closest_from_right_1st,rows),(0,0,255),1)

            cv2.line(evaluation_img,(closest_from_left_2nd,0),(closest_from_left_2nd,rows),(0,0,255),1)
            cv2.line(evaluation_img,(closest_from_right_2nd,0),(closest_from_right_2nd,rows),(0,0,255),1)

    else:
        img_split_vertically = img_split_horizontally


    if show == True:
        for i in img_split_vertically:
            show_image(i)
        show_image(evaluation_img)


    return img_split_vertically, evaluation_img

  