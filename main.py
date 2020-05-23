from imutils import paths
import numpy as np
import argparse
import cv2
import logging
import sys
import matplotlib.pyplot as plt

def stitch_images_CV(images, output):
  stitcher = cv2.Stitcher_create()
  (status, stitched) = stitcher.stitch(images)
  
  if status == 0:
    cv2.imwrite(output, stitched)

def stitch_images(images, output):
  detect_features(images)

  pass

def detect_features(images):
  orb = cv2.ORB_create()
  # features = []
  # for img in images:
  #   features.append(orb.detectAndCompute(img, None))
  img1 = images[0]
  img2 = images[1]
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)

  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  # Match descriptors.
  matches = bf.match(des1,des2)
  #print(matches)
  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)
  # Draw first 10 matches.
  img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  plt.imshow(img3),plt.show()


if __name__ == "__main__":
  logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
  parser = argparse.ArgumentParser(description="Panoramic Image Stitching")
  parser.add_argument("inputImages", type=str, help="Input image directory")
  parser.add_argument("outputImage", type=str, help="Output image")
  parser.add_argument("--crop", help='Auto-crop the output image', action="store_true")
  parser.add_argument("--blend", help='Blend the images', action="store_true")
  args = parser.parse_args()

  inputPath = sorted(list(paths.list_images(args.inputImages)))
  images = []
  output = args.outputImage

  for i in inputPath:
    image = cv2.imread(i)
    images.append(image)

  #stitch_images_CV(images, output)

  stitch_images(images, output)