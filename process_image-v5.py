# -*- coding: utf-8 -*-

"""

This script:

- reads in an image file containing droplets (and beads within)
- does some background enhancement 
(depends, some images need this and thus, droplet detection works better; 
check the intermittent images and droplet-detection results by the eye;
activate the plotting funtion for this visual examination)

- does edge detection 
(depends, some images need this and thus, droplet detection works better; 
check the intermittent images and droplet-detection results by the eye;
activate the plotting funtion for this visual examination)

- labels regions of droplets by using rectangles
- prints out total number of rectangles detected aka droplets
- detects the beads within droplets 

...


To-Do (features):
- count the beads
- ask the user if he wants to use background enhancement yes/no
- ask the user if he wants to do edge detection yes/no
    

"""

import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage import data, io, filters, img_as_float
from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
 


def findtemplate(templateimagefile, targetimagefile, cropdim, outfile, thresholdval):
    ##
    ## why am I reading files here?
    ## can possibly just pass the cropped-image from where this function was called
    ##
    ## naive reason that this function is being developled from an independent python script that worked for the whole image
    ## want to slowly change things and make them work for the cropped-subset of an image
    ## if all goes well can try change this and not read files here but just take cropped image passed from the calling function
    ## for now, taking what works over efficience and elegance
    ## 
    ## scikit-image and open cv dont work together
    ## need a simple format conversion
    ## http://scikit-image.org/docs/dev/user_guide/data_types.html#working-with-opencv
    ##
    ##
    ## this is taken from internet
    ## look here under: "Template Matching with Multiple Objects'
    ## http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html 
    ##

    img_rgb = cv2.imread(targetimagefile)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(templateimagefile,0)
    minr, minc, maxr, maxc = cropdim
    crop_image_rgb = img_rgb[minr:maxr, minc:maxc]
    #io.imshow(crop_image_rgb)   ## check the image
    #io.show()   ## check the image
    crop_image_gray = img_gray[minr:maxr, minc:maxc]
    #io.imshow(crop_image_gray)   ## check the image
    #io.show()   ## check the image
    
    w, h = template.shape[::-1]
    img_gray=crop_image_gray
    img_rgb=crop_image_rgb 
    
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = thresholdval
    loc = np.where( res >= threshold)
    rowcoord=[]
    colcoord=[]

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        rowcoord.append(pt[1])
        colcoord.append(pt[0])


    if len(rowcoord) <1:
        n_clusters_ = 0
    else:
        X=np.matrix(zip(rowcoord,colcoord))
        
        from sklearn.cluster import AffinityPropagation
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        af = AffinityPropagation(preference=-50).fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        
        try:
            n_clusters_ = len(cluster_centers_indices)
            """
            from itertools import cycle
            plt.close('all')
            plt.figure(1)
            plt.clf()
            colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
            for k, col in zip(range(n_clusters_), colors):
                class_members = labels == k
                cluster_center = X[cluster_centers_indices[k]]
                plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
            plt.title('Estimated number of beads: %d' % n_clusters_)
            plt.show()
            """
            
        except TypeError: 
                n_clusters_ = 0


    totalbeads =  n_clusters_
    print "...Total beads detected: %d" % totalbeads 
    fileout= ''.join([path.splitext(targetimagefile)[0], '_droplet-', str(outfile), '_totalbeads-', str(totalbeads), '.png'])
    print fileout
           
    cv2.imwrite(fileout,img_rgb)
        
    return



from os import path 
import argparse
def main():
    parser=argparse.ArgumentParser(description='Image-file processing to identify total number of droplets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_image', type=str, 
                      help='input image file')
    parser.add_argument('template_image', type=str, 
                      help='template image file')
    parser.add_argument('--outfile', type=str,
                      help='output image name.  Will default to input_image_processed.png')
    parser.add_argument('--cutoff_area', type=int,
                      help='rectangle size. only detected rectangles above this size will be displayed')
    parser.add_argument('--include', action='store_true', dest="include_edge", 
                      help='Include edge or boundary or border droplets; incomplete droplets found on the edge of the picture will be included too.')
    parser.add_argument("--exclude", action="store_false", dest="include_edge",
                      help='Exclude edge or boundary or border droplets; incomplete droplets found on the edge of the picture will be excluded.')
    parser.add_argument('--threshold', type=float, default=0.6,
                      help='thresholding parameter, tweak from 0 to 1 and visually examine results, depends on the image, default is 0.6')


    args=parser.parse_args()
    
    if args.outfile==None:
        args.outfile=''.join([path.splitext(args.input_image)[0],"_border_",str(args.include_edge),'_processed']) ## leaving out the file extension; adding it later just before writing out the image file

    filename=args.input_image
    rectsize=args.cutoff_area


    print "...."
    print "image file being processed..."
    print "..."
    
   
    from skimage import img_as_float
    image = io.imread(filename, flatten=True) # conver 3d to 2d image by using flatten=True
    image = img_as_float(image)
    #io.imshow(image)   ## check the image
    #io.show()   ## check the image

    from skimage.color import rgb2gray
    img_gray = rgb2gray(image)
    #io.imshow(img_gray)   ## check the image
    #io.show()   ## check the image
    
    image = gaussian_filter(image, 1)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    
    dilated = reconstruction(seed, mask, method='dilation')
    

    """    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                        ncols=3,
                                        figsize=(8, 2.5),
                                        sharex=True,
                                        sharey=True)
    
    ax0.imshow(image, cmap='gray')
    ax0.set_title('original image')
    ax0.axis('off')
    ax0.set_adjustable('box-forced')
    
    ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
    ax1.set_title('dilated')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    
    ax2.imshow(image - dilated, cmap='gray')
    ax2.set_title('image - dilated')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    
    fig.tight_layout()
    """

    
    print "...."
    print "background correction..."
    print "..."
    
    h = 0.4
    seed = image - h
    dilated = reconstruction(seed, mask, method='dilation')
    hdome = image - dilated
    #io.imshow(hdome)   ## check the image
    #io.show()   ## check the image

    
    """
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))
    yslice = 197
    
    ax0.plot(mask[yslice], '0.5', label='mask')
    ax0.plot(seed[yslice], 'k', label='seed')
    ax0.plot(dilated[yslice], 'r', label='dilated')
    ax0.set_ylim(-0.2, 2)
    ax0.set_title('image slice')
    ax0.set_xticks([])
    ax0.legend()
    
    ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
    ax1.axhline(yslice, color='r', alpha=0.4)
    ax1.set_title('dilated')
    ax1.axis('off')
    
    ax2.imshow(hdome, cmap='gray')
    ax2.axhline(yslice, color='r', alpha=0.4)
    ax2.set_title('image - dilated')
    ax2.axis('off')
    
    fig.tight_layout()
    plt.show()
    """    
    
    
    print "...."
    print "edge detection..."
    print "..."
    
    im = hdome
    edges1 = feature.canny(image, sigma=3)
    edges2 = feature.canny(im, sigma=3)
    #io.imshow(edges1)   ## check the image
    #io.show()   ## check the image
    #io.imshow(edges2)   ## check the image
    #io.show()   ## check the image



    """
    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('Original image', fontsize=10)
    
    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter on original image, $\sigma=3$', fontsize=10)
    
    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter on background subtracted image, $\sigma=3$', fontsize=10)
    
    fig.tight_layout()
    plt.show()
    """    
        

 
#    
### check how good are the original and processed images by selecting corresponding image here
#   
    image=image
    #image=edges2
    #image=hdome
    
    ## apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(2))
    #io.imshow(bw)   ## check the image
    #io.show()   ## check the image
    
    
    print "... are we including incomplete droplets at the edge of image?..."
    print ".............................................................", args.include_edge
    if args.include_edge is False:
        ## remove artifacts connected to image border
        cleared = clear_border(bw)    ## use this option to avoid the incomplete droplets at boundary/edge of picture frame
    else:
        cleared = bw  ## use this to use all droplets in the image; even incomplete ones at the boundary/edge of pciture frame
    
    
    #io.imshow(cleared)   ## check the image
    #io.show()   ## check the image
        
    
    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)
    #io.imshow(image_label_overlay)   ## check the image
    #io.show()   ## check the image
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.imshow(label_image)
    #ax.imshow(image_label_overlay)


    targetimagefile=args.input_image
    templateimagefile=args.template_image
    thresholdval=args.threshold

    beads_count=0
    droplet_count=0
    outfile=0
    for region in regionprops(label_image):
        # take regions with large enough areas; should correspond to droplets        
        if region.area >= rectsize:
            # draw rectangle around segmented droplets
            droplet_count=droplet_count+1
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='yellow', linewidth=2)
            #print region.bbox
            try:
                crop_image = image[minr-50:maxr+50, minc-50:maxc+50]  ## offset the bounding box in all directions; seems better this way based on visual examination
            except ValueError:  #raised if this subtraction takes it out of bounds
                pass
            finally:
                crop_image = image[minr:maxr, minc:maxc]
            #io.imshow(crop_image)   ## check the image
            #io.show()   ## check the image
            outfile=outfile+1
            ## improve this and instead of passing file names of whole images to the findtemplate function
            ## pass the cropped image that already exists here
            ## for now, not changing too drastically on something that works
            ## slow tweaks ensuring what works functionally well doesnt break down
            beads=findtemplate(templateimagefile, targetimagefile, region.bbox, outfile, thresholdval)
        
            ax.add_patch(rect)
              
    ax.set_axis_off()
    plt.tight_layout()
    outfile=args.outfile + "_totaldroplets-" + str(droplet_count) + ".png"
    print "...saving image file...."
    print outfile
    print "........................"
    plt.savefig(outfile)
    #plt.show()      ## activate this if you want to examine how the processed images are turning out and stop/start with different input parameters; key one being --cutoff_area

 
    
    print "...total droplets identified in the image:"
    print droplet_count
    print "..."
    
    

if __name__ == '__main__':
    main()

    
    




