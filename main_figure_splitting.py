from fig_splitting import *

import cv2
import glob
import random


def split_figure(figname, eval_fig = False):
    '''Splits figures mined from publications into their constituent images. Note: Must be used on the 
    products of FDE's figure splitting process.

    :param string figname: Name of the input figure.
    :param bool eval_fig: Optionally output an annotated version of the input for evaluation.

    :return list fig_split_final: list of constituent images as numpy.ndarrays.
    '''

    fig = cv2.imread(figname)

    #fig already grayscale.
    if len(fig.shape)==2:
        gfig=fig
    #fig is in color, convert to grayscale.
    else:
        gfig=cv2.cvtColor(fig,cv2.COLOR_BGR2GRAY)

    #Splitting process.
    fig_split_final, evaluation_fig = line_detection_and_split(gfig,eval_img = eval_fig)
    
    #Optional writing of images.
    if fig_split_final is not None and eval_fig == True:

        cv2.imwrite("eval"+ "_" +str(figname).split("/")[-1],evaluation_fig)    

        index = 0
        for fig in fig_split_final:
            cv2.imwrite(str(index) + "_" +str(figname).split("/")[-1],fig)
            index += 1


    return fig_split_final






imgnamelist=[]
path = "/Users/karimmukaddem/Desktop/imagedataextractor/eval_sem_tio2_nano/1_matt_split/*.png"

imgnamelist.extend(glob.glob(path))


for imgname in imgnamelist:

    if str(str(imgname).split("/")[-1])[0] != "e":

        print str(str(imgname).split("/")[-1])

        split_figure(imgname, True)


