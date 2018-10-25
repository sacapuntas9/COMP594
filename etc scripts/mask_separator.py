import cv2
import numpy as np
import os
import sys

directorySeparator = '\\'


inputDir = "E:\\594_data\\18_scan_duplicate_removed\\combined_masks"
outputPeri = "E:\\594_data\\18_scan_duplicate_removed\\masks_pericyte"
outputSmooth = "E:\\594_data\\18_scan_duplicate_removed\\masks_smooth_muscle"
outputBoth = "E:\\594_data\\18_scan_duplicate_removed\\masks_both"

directory = os.fsencode(inputDir)

for filename in os.listdir(directory):
    file = os.fsdecode(filename)
    if file.endswith(".tif"): 
        image = cv2.imread(inputDir + directorySeparator + file, -1)
        
        imagePeri = image
        imagePeri= (imagePeri/256).astype('uint8') #convert to 8 bit
        imageSmooth = image
        imageSmooth = (imageSmooth/256).astype('uint8') #convert to 8 bit
        imageBoth = image
        imageBoth = (imageSmooth/256).astype('uint8') #convert to 8 bit
        
        h = image.shape[0]
        w = image.shape[1]
        
        for y in range(0, h):
            for x in range(0, w):
                # create the masks
                imagePeri[y, x] = 255 if image[y, x] == 2 else 0
                imageSmooth[y, x] = 255 if image[y, x] == 1 else 0
                imageBoth[y, x] = 255 if (image[y, x] == 1 or image[y, x] == 2)  else 0
        
        cv2.imwrite(outputPeri + directorySeparator + file,imagePeri)
        cv2.imwrite(outputSmooth + directorySeparator + file,imageSmooth)
        cv2.imwrite(outputBoth + directorySeparator + file,imageBoth)
        