from imutils import paths
import imutils
import numpy as np
import cv2

def existing_algorithm(img, output):
	new_img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	mask = np.zeros(thresh.shape, dtype="uint8")
	(x, y, w, h) = cv2.boundingRect(c)
	rect = cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

	minRect = mask.copy()
	sub = mask.copy()

	while cv2.countNonZero(sub) > 0:
		minRect = cv2.erode(minRect, None)
		sub = cv2.subtract(minRect, thresh)

	cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)

	result = img[y:y + h, x:x + w]
	cv2.imwrite(output, result)
			 

	