# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:25:24 2017

This script can detect beads in the image.
It does by using "template matching".
An image of bead is used as the template to look for in the whole image (details below.
Tweak threshold values to get suitable results (details below).

USE A TEMPLATE IMAGE:
For detecting beads, I crop out a bead-image from one of the files, and use it as a template to find the other beads in the whole-image (‘template-matching’).
A bead-image taken from any one file works quite well on any of the other image files.
Pragmatically, the image-quality, look and feel of the template bead image should match that of the target images where we are detecting the beads.

THRESHOLD VALUE:
There is a threshold parameter to tweak from 0 to 1. I have set a default value of 0.6 based on some experimentation.
How well this parameter has worked is easily examined from a quick look at the resulting processed image.
It effectively sets signal/noise ration of beads detected.

@author: dogrask
"""


import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')


import cv2
import numpy as np
from matplotlib import pyplot as plt


from os import path 
import argparse
def main():
    parser=argparse.ArgumentParser(description='Image-file processing to identify total number of beads',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_image', type=str, 
                      help='input image file')
    parser.add_argument('template_image', type=str, 
                      help='template image file')
    parser.add_argument('--threshold', type=float, default=0.6,
                      help='thresholding parameter, tweak from 0 to 1 and visually examine results, depends on the image, default is 0.6')
    parser.add_argument('--outfile', type=str,
                      help='output image name.  Will default to inputimage_thresholdval_processed.png')
    args=parser.parse_args()
    
    if args.outfile==None:
        args.outfile=''.join([path.splitext(args.input_image)[0],"_threshold",str(args.threshold),'_processed.png'])


    filename=args.input_image
    templatefile=args.template_image
    thresholdval=args.threshold

    print "...."
    print "image file being processed..."
    print "..."
    

##
## this is taken from internet
## look here under: "Template Matching with Multiple Objects'
## http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html 
##
##
    
    img_rgb = cv2.imread(filename)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(templatefile,0)
    w, h = template.shape[::-1]
        
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = thresholdval
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    cv2.imwrite(args.outfile,img_rgb)


if __name__ == '__main__':
    main()

