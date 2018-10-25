import cv2
import numpy as np
import os
import sys

directorySeparator = '\\'


inputDir = "E:\\594_data\\combined_ground_truth\\images"
outputDir = "E:\\594_data\\NeuralNetData\\Combined\\images"


directory = os.fsencode(inputDir)

for filename in os.listdir(directory):
    file = os.fsdecode(filename)
    if file.endswith(".tif"): 
        image = cv2.imread(inputDir + directorySeparator + file, -1)
        
        image= (image/256).astype('uint8') #convert to 8 bit
        
        cv2.imwrite(outputDir + directorySeparator + file+".png",image)
        