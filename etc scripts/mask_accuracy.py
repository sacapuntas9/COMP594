import cv2
import numpy as np
import os
import sys
from skimage.measure import structural_similarity as ssim

directorySeparator = '\\'

generatedMaskDir = "E:\\594_data\\90_manually_marked_with_auto\\combined_masks"
groundTruthDir = "E:\\594_data\\90_manually_marked_with_auto\\masks_pericyte"


directoryMask = os.fsencode(generatedMaskDir)
directoryTruth = os.fsencode(groundTruthDir)




def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

for filename in os.listdir(directoryMask):
    file = os.fsdecode(filename)
    if file.endswith(".tif"): 
        imageMask = cv2.imread(generatedMaskDir+ directorySeparator + file, -1)
        imageTruth = cv2.imread(groundTruthDir+ directorySeparator + file, -1)
        
        m = mse(imageMask,imageTruth )
        s = ssim(imageMask,imageTruth )
        
       print("MSE :"+m)
       print("SSIM :"+ssim)
        
        
