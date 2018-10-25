from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import os
import sys
import time
import multiprocessing


#NOTE: If you are running on a system that has different directory separators than windows, change them here.
directorySeparator = '\\'


def generate_mask(queue,inputDir,outputDir): #function that is run by each slave process in the process pool
	while not queue.empty(): #check if queue is empty
		start = time.time()
		filename = queue.get() #if queue is not empty, pop the next task off of the queue (file, in this case)
		if filename.endswith(".tif"): 
			print("Beginning to process " + filename + ". Current time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

			image = cv2.imread(inputDir + directorySeparator + filename, -1) #read the input image from disk
			
			converted = (image/256).astype('uint8') #convert input to 8 bit
			
			thresh = cv2.threshold(converted, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #auto thresholding using otsu method
			#documentation: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
			#the 100 and 255 are normally the range for the thresholding, but when THRESH_OTSU is used, these values are ignored because they are detected algorithmically.
			
			
			
			
			# perform a connected component analysis on the thresholded
			# image, then initialize a mask to store only the desired
			# components

			labels = measure.label(thresh, neighbors=8, background=0)
			#generate a collection of connected regions, using any amount of shared vertices to define a connection, with a background value of 0
			#labels contains an array with the same shape as the original image, but with unique connected components all sharing a unique number value
			#documentation: http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
			
			
			mask = np.zeros(thresh.shape, dtype="uint8")
			#generate an array with the same shape as the thresholded image, with all zeroes. Used to add unique connected components of desired parameters.
			#https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.zeros.html

			# loop over the unique components
			for label in np.unique(labels):
				# if this is the background label, ignore it
				if label == 0:
					continue
			 
				# otherwise, construct the label mask and count the
				# number of pixels 
				
				labelMask = np.zeros(thresh.shape, dtype="uint8")
				#create another blank array with the shape of the image
				
				labelMask[labels == label] = 255
				#set every value in the labelMask array to 255 where the location of the currently iterated label is in the connected regions array
				#This step converts the unique value in the labels array to 255, generating a binary mask containing the unique component
				
				numPixels = cv2.countNonZero(labelMask)
				#count the number of pixels in the unique component
			 
				# if the number of pixels in the component is of desired parameters,
				# add it to the final mask array
				if (numPixels > 5 and numPixels < 750):
					mask = cv2.add(mask, labelMask)

			cv2.imwrite(outputDir + directorySeparator + filename,mask) #write the final mask to disk
			end = time.time()
			print("----->time taken for " + filename + ": "+str(end - start) + " seconds" )

		
		
		
#The following code is run only on the original, first process, and is run when the program starts
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Takes directory input, iterates through .tif files in directory, processes binary mask, and outputs to output directory.')
	parser.add_argument("-i", "--input", required=True,
	help="Input Directory")
	
	parser.add_argument("-o", "--output", required=True,
	help="Output Directory")

	parser.add_argument("-p", "--processes", required=True,
	help="Maximum number of concurrent processes")
	
	args = vars(parser.parse_args())

	inputDir = args["input"]
	outputDir = args["output"]
	numProcesses = args["processes"]   #read in arguments
	
	queue = multiprocessing.Queue()  #use multiprocessing queue to queue tasks 
	
	
	directory = os.fsencode(inputDir) #encode input directory so it can be iterated

	for file in os.listdir(directory):#insert all files from input directory into 
		queue.put(os.fsdecode(file))
	
	
	print("Beginning processing with "+numProcesses+" processes.")
	pool = multiprocessing.Pool(int(numProcesses), generate_mask,(queue,inputDir,outputDir,))  #pass the number of processes, the function, and the args for the function to be run with those processes
	pool.close() # signal that we won't submit any more tasks to pool
	pool.join() # wait until all processes are done
	print("Image processing complete.")






	
		







