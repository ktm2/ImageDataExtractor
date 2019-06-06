import cv2
import numpy as np
import keras.backend as K
import os
from  PIL import Image


def writeout_image(img, output_path, filteredvertices, imgname, inverted = False):
    '''Output an image displaying all the vertices detected.
    :param string outputpath: path to output directory.
'''

    drawing_img = crop_image(img.copy())

    for i in range(len(filteredvertices)):
        cv2.polylines(drawing_img,[filteredvertices[i]],True,(0,255,0),thickness=1)
        #Annotate particle #.
        annotate = True
        if annotate == True:
            (xcom,ycom),contradius = cv2.minEnclosingCircle(filteredvertices[i])
            xcom=int(xcom)
            ycom=int(ycom)
            contradius=int(contradius)
            cv2.circle(drawing_img,(xcom,ycom),1,(0,0,255),1)
            cv2.putText(drawing_img,str(i+1),(xcom+3,ycom+3),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,255),thickness=1)           

    if inverted == True:
        cv2.imwrite(os.path.join(output_path, "inv_det_"+str(imgname).split("/")[-1]),drawing_img)
    else:
        cv2.imwrite(os.path.join(output_path, "det_"+str(imgname).split("/")[-1]),drawing_img)


    return

def crop_image(img, crop_constant = 2):
    '''Crop image in slightly, helps with splitting artifacts. 

    :param numpy.ndarray img: the image to be framed.
    :param int crop_constant: number of pixels to eat in.

    :return numpy.ndarray img: cropped img.
    '''

    rows = len(img)
    cols = len(img[0])

    img = img[crop_constant:rows-crop_constant, crop_constant:cols-crop_constant]

    return img


def frame_image(img, frame_constant = 3):
    '''Place a frame around the edges of the image, does not grow
    instead replaces outermost pixels.

    :param numpy.ndarray img: the image to be framed.
    :param int frame_constant: number of pixels to eat in.

    :return numpy.ndarray img: cropped img.
    '''

    rows = len(img)
    cols = len(img[0])

    cv2.rectangle(img,(0,0),(frame_constant,rows),(0,0,0),-1)
    cv2.rectangle(img,(0,rows-frame_constant),(cols,rows),(0,0,0),-1)
    cv2.rectangle(img,(0,0),(cols,frame_constant),(0,0,0),-1)
    cv2.rectangle(img,(cols-frame_constant,0),(cols,rows),(0,0,0),-1)

    return img


def ask_to_save_image(img, imgname):
    '''Asks user if current image should be saved. Useful for evaluation

    :param numpy.ndarrray img: input image to be saved
    :param string imgname: desired name of filename

    '''

    tosaveornottosave = raw_input("Save image (y/n)? : ")

    if tosaveornottosave == "y" or tosaveornottosave == "Y":
        cv2.imwrite(imgname,img)

    return


def image_metrics(gimg,displaymetrics=False):
    '''Calculate image metrics and optinally display it.
    :param numpy.ndarray gimg: grayscale input image.
    :param bool displaymetrics: prints calculated values and displays image, useful in debugging.

    :return int rows: number of rows
    :return int cols: number of columns
    :return int imgarea: pixels in image
    :return int imgmean: mean pixel intensity in image
    :return float imgstdev: standard deviation of pixel intensities
    :return float crossstdev: standard of pixel intensities across image diagonals. 

    '''

    rows = len(gimg)
    cols = len(gimg[0])
    flatgimg = gimg.flatten()
    imgarea = rows*cols
    imgmean = sum(flatgimg)/len(flatgimg)
    imgstdev = np.std(flatgimg)
    crossstdev = np.std(np.diagonal(gimg))

    if displaymetrics == True:
        print("mean: " + str(imgmean))
        print("stdev: " + str(imgstdev))
        print("area: " + str(imgarea))
        print("crossstdev: " + str(crossstdev))
        plt.hist(gimg.ravel(),256,[0,256])
        plt.show()

    return rows, cols, imgarea, imgmean, imgstdev, crossstdev


def show_image(imgfilename,waitkey = 0,imgname = None,mag = 2):
    '''Used to display images in cv2 format
    :param numpy.ndarray imgfilename: input image.
    :param int waitkey: miliseconds to display image, if 0 user must close manually
    :param string imgname: header while displaying image.
    :param int mag: amount to magnify image by.

    '''

    if imgname == None:
        imgname = str(imgfilename)

    if mag != None:
        imgfilename = cv2.resize(imgfilename,None,fx = mag, fy = mag)

    cv2.imshow(imgname,imgfilename)
    cv2.moveWindow(imgname,0,0)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()
    
    return


def distance_formula(a,b):
    '''Inputs two coordinate points (x,y), outputs distance between them
    :param tuple a: (x,y) coordinate of 1st point
    :param tuple b: (x,y) coordinate of 2nd point

    :return int d: distance between two points
    '''

    d=((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

    return d


def psnr(y_true, y_pred):
    diff = y_pred - y_true
    diff = K.flatten(diff)
    rmse = K.sqrt(K.mean(diff ** 2.))
    log10 = K.log(255.0/rmse) / K.log(10.0)
    return 20.0 * log10


def convert_gif_to_png(imgname, secondary):
    """Converting GIFs to PNG"""

    new_imgname = imgname.rsplit('.', 1)[0] + '.png'
    temp_gif = Image.open(imgname)
    temp_gif.save(new_imgname)

    os.remove(imgname)

    secondary = [v.rsplit('.')[0] + '.png' for v in secondary]

    return new_imgname, secondary

