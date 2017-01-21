# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 21:59:49 2016

@author: nishan
"""

# import the necessary packages
import numpy as np
import cv2
import imutils
import glob
 
def process(filename, key):
    
    # load the image
    image = cv2.imread(filename,0)
    
    #Apply median filter to remove noise
    median = cv2.medianBlur(image,3)

    #Normalise the image
    cv2.normalize(image, image,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F ) 
    
    #Sharpened using unsharp masking method
    gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

    #Edge detection using canny edge detection
    th, bw = cv2.threshold(unsharp_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(unsharp_image, th/2, th)
    
    #cv2.imshow("sharp", unsharp_image)
    #cv2.imshow("median", image)
    
    #th, bw = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = 150
#    print(unsharp_image.shape)
#    
    height,width = unsharp_image.shape
#    
    lw= int(width * 0.12 )
    rw = int(width * 0.6)
    uh = int(height * 0.17)
    dh = int(height * 0.81)
    ROI0 = unsharp_image[uh:dh, lw: rw]#Remove skull and neck area
#    
#    wid,hei = ROI0.shape
    #print(ROI0.shape)
    
#    black_img = np.zeros_like(unsharp_image)
    #cv2.imshow("black_img", black_img)
    #rect1 = np.array([ [120,0], [5,3], [5,30], [120, 30] ], np.int32)#Cortex
    #rect2 = np.array([ [120,76], [5,76], [5,106], [120, 106] ], np.int32)#Cortex
    #rect3 = np.array([ [68,30], [40,30], [40,75], [68, 75] ], np.int32)#Ventricles
    #rect4 = np.array([ [105,30], [85,30], [85,45], [105, 45] ], np.int32)#Hippacampus
    #rect5 = np.array([ [105,62], [85,62], [85,75], [105, 75] ], np.int32)#Hippacampus
    #rect6 = np.array([ [25,30], [5,30], [5,76], [25, 76] ], np.int32)#Cortex
    
    #cv2.fillPoly(black_img, [rect1], (255,255,255))
    #cv2.fillPoly(black_img, [rect2], (255,255,255))
    #cv2.fillPoly(black_img, [rect3], (255,255,255))
    #cv2.fillPoly(black_img, [rect4], (255,255,255))
    #cv2.fillPoly(black_img, [rect5], (255,255,255))
    #cv2.fillPoly(black_img, [rect6], (255,255,255))
    #cv2.imshow("ROI", black_img)
    
    #print(black_img.shape)
    #new_img = cv2.bitwise_and(ROI0,black_img, black_img)
    #RO = black_img[10:50, 20:60]
    #cv2.imshow("RO", new_img)
    #ROI1 = unsharp_image[60:105, 58: 106 ]
    #cv2.imshow("edges", ROI0)
    #cv2.imshow("edges1", edges)
    #return new_img
    return edges

for (i,image_file) in enumerate(glob.iglob('Extended/*.jpg')):
        name = str(image_file)[9:]
        print(name)
        img = process(image_file, i)
        cv2.imwrite("CannyExtended/" +name, img)
#cv2.waitKey(0)
