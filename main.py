from imutils import paths
import numpy as np
import argparse
import cv2
import logging
import sys

def stitch_images(images, output):
	"""
	Image stitching (Panorama) using OpenCV's stitcher.

	args:
		images: a list of images to stitch
		output: a name of output
	returns:
		Stitched image 
	"""
	stitcher = cv2.Stitcher_create()
	(status, stitched) = stitcher.stitch(images)
	
	if status == 0:
		cv2.imwrite(output, stitched)
		return stitched

def create_mask(img):
	"""
	Create a mask. 0 represents non-black pixels,
	255 represents blakc pixels
	args:
		img: stitched image from stitch_image method
	returns:
		mask: a mask of the image
	"""

	# Convert the image to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)[1]
	return mask

def filling_pixels(img, output):
	"""
	Fill black pixels on the edges of the stitched image with
	OpenCV's inpaint method.
	Use both Navier-Stoeks and Telea's algorithm
	args:
		img: stitched image from stitch_image method
		ouput: a name of output
	returns:
		Saves inpainted images with the output name given
	"""
	
	# Find a mask
	mask = create_mask(img)

	# Apply inpaint to fill the black pixels
	filled_NS = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
	cv2.imwrite("NS_filled" + output, filled_NS)
	filled_TELEA = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
	cv2.imwrite("TELEA_filled" + output, filled_TELEA)

def largest_rectangle(arr):
	"""
	Find the largest rectangle which only contains 0's
	in a given array which represents a histogram
	args:
		arr: a numpy array representing the histogram
	returns:
		max_area: maximum rectangular area under histogram
		max_left: left index returning the largest rectangle
		max_right: right index returning the largest rectangle
		max_height: height returning the largest rectangle
	"""
	stack = []

	left_indices = []
	right_indices = []

	for i  in range(len(arr)):
		while stack and arr[stack[-1]] >= arr[i]:
			stack.pop()

		left_indices.append(-1 if not stack else stack[-1])
		stack.append(i)

	stack = []

	for i in range(len(arr) - 1, -1, -1):
		while stack and arr[stack[-1]] >= arr[i]:
			stack.pop()

		right_indices = [len(arr) if not stack else stack[-1]] + right_indices
		stack.append(i)

	max_area = 0
	max_right = 0
	max_left = 0
	max_height = 0

	for i in range(len(arr)):
		# Find the area of the rectangle at a given index
		temp_area = arr[i] * (right_indices[i] - left_indices[i] - 1)
		# Update max variables when found larger rectangle
		if temp_area > max_area:
			max_area = temp_area
			max_right = right_indices[i]
			max_left = left_indices[i]
			max_height = arr[i]

	return max_area, max_left, max_right, max_height

def largest_2d_array(mask):
	"""
	Find the largest rectangle that only contains 0's
	for a given mask
	args:
		mask: a mask returned from create_mask method
	returns:
		max_left: left index returning the largest rectangle
		max_right: right index returning the largest rectangle
		max_height: height returning the largest rectangle
		max_y: y coordinate returning the largest rectangle
	"""

	# Initialize a numpy array the same size as the mask, containing all 0s
	max_height_table = np.zeros(dtype=int, shape=mask.shape)

	# Iterate through the mask row by row from the bottom upward to find
	# maximum height that only contains 255 continuously 
	for i in range(len(mask) - 1, -1, -1):
		if i == len(mask) - 1:
			max_height_table[len(mask) - 1] = mask[-1]
			continue

		for j, column in enumerate(mask[i]):
			if column == 255:
				continue
			max_height_table[i][j] = max_height_table[i + 1][j] + 1

	max_area = 0
	max_left = 0
	max_right = 0
	max_height = 0
	max_y = 0

	for i in range(len(mask)):
		# Find the largest rectangular area each row
		largest_subarray_area, idx_left, idx_right, height = largest_rectangle(max_height_table[i])
		# Update the max variables when larger area is found
		if largest_subarray_area > max_area:
			max_area = largest_subarray_area
			max_left = idx_left
			max_right = idx_right
			max_height = height
			max_y = i
	
	# When left index is -1, set it to 0
	max_left = max_left + 1 if max_left == -1 else max_left

	return max_left, max_right, max_height, max_y
	
def auto_cropped(img, output):
	"""
	Cropping the stitched image to remove black pixels
	args:
		img: stitched image from stitch_images method
		output: a name of output
	returns:
		Save cropped image to the output file with a prefix 'cropped_'
	"""

	# Add few pixels around the stithced image
	new_img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

	# Creat a mask
	mask = create_mask(new_img)

	max_left, max_right, max_height, max_y = largest_2d_array(mask)

	# Get the cropped image using indices, starting y value and height
	result = new_img[max_y:max_y + max_height, max_left: max_right, :]
	cv2.imwrite("cropped_" + output, result)

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
	parser = argparse.ArgumentParser(description="Panoramic Image Stitching")
	parser.add_argument("inputImages", type=str, help="Input image directory")
	parser.add_argument("outputImage", type=str, help="Output image")
	parser.add_argument("--crop", help='Auto-crop the output image', action="store_true")
	parser.add_argument("--fill", help='Fill the black pixels', action="store_true")
	args = parser.parse_args()

	inputPath = sorted(list(paths.list_images(args.inputImages)))
	images = []
	rotated_images = []
	output = args.outputImage

	for i in inputPath:
		image = cv2.imread(i)
		images.append(image)

	stitched = stitch_images(images, output)

	if args.crop:
		auto_cropped(stitched, output)
	
	if args.fill:
		filling_pixels(stitched, output)
			 