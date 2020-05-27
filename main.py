from imutils import paths
import numpy as np
import argparse
import cv2
import logging
import sys
import matplotlib.pyplot as plt

def stitch_images(images, output):
  stitcher = cv2.Stitcher_create()
  (status, stitched) = stitcher.stitch(images)
  
  if status == 0:
    cv2.imwrite(output, stitched)
    filling_pixels(stitched)

def filling_pixels(img):
  # Find the black pixels
  temp = (img == 0)

  # Create a mask
  mask = temp.astype(np.uint8) * 255
  mask_gray = np.mean(mask, axis=2)
  mask_gray = mask_gray.astype(np.uint8)

  # Apply inpaint to fill the black pixels
  filled = cv2.inpaint(img, mask_gray, 3, cv2.INPAINT_TELEA)
  cv2.imwrite("filled.jpg", filled)


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

  stitch_images(images, output)