from .scale_reading import *


def scalebar_identification(img, outputpath='', testing = None):
    '''Runs scalebar detection on SEM images.
    :param numpy.ndarray img: input image..
    :param string testing: Name of evaulation image for output.
    
    :return float scale: Distance associated with a pixel in units of (m/pixel)
    :return list inlaycoords: list of tuples, (x,y,w,h) top left corner, width and 
    height of inlays, including scalebar.
    :return float conversion: unit of scalevalue 10e-6 for um, 10e-9 for nm.


    TODO:
    - Inlay determination.
    
    '''

    scale, scalebar, scalevalue, conversion  = None, None, None, None

    inlaycoords = []

    #Check if img already grayscale.
    if len(img.shape) == 2:
        gimg = img
    #If img is in color, convert to grayscale.
    else:
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Calculate image metrics.
    rows, cols, imgarea, imgmean, imgstdev, crossstdev = image_metrics(gimg)

    #Perform idenfitication.
    scalebar, scalevalue, conversion, inlaycoords = detect_scale_bar_and_inlays(gimg, imgmean,
     imgstdev, rows, cols)



    #Optinally output an image for evaulation purposes.
    if testing != None:
        output_img = img.copy()
        if scalebar != None:
            cv2.rectangle(output_img,(scalebar[0],scalebar[1]),(scalebar[0]+scalebar[2],scalebar[1]+scalebar[3]),
            (0,0,255),thickness=1)
        if len(inlaycoords) > 0:
            for box in inlaycoords:
                cv2.rectangle(output_img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),
            (255,0,0),thickness=1)

        cv2.putText(output_img, str(scalevalue)+"*"+str(conversion), (cols//2,rows//2),cv2.FONT_HERSHEY_PLAIN,1, (0,0,255),thickness=1)
        cv2.imwrite(os.path.join(outputpath, "scalebar_"+str(testing).split("/")[-1]),output_img)

    if scalebar != None:
        scale = scalevalue * conversion / float(scalebar[2])

    return scale, inlaycoords, conversion
