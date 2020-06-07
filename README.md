# Auto-Cropped Panorama Image Stitching
This program stitches individual images to create a single panoramic image.  Without arguments, the program will simply stitch the images nad return the generated result.  Using the argument ```--fill``` will fill in black pixels within the panoramic result.  Using the argument ```--crop``` will remove the black pixels within the panoramic result.
 
## Getting Started

### Prerequisites
This program will only run on Python 3

### Installing
* ```pip install imutils```
* ```pip install numpy```
* ```pip install argparse```
* ```pip install opencv-python```

## Usage

1. Basic Image Stitching

```python main.py <Input Image Directory> <Output Image>```

2. Filling Black Pixels

```python main.py --fill <Input Image Directory> <Output Image>```

3. Auto-Cropping

```python main.py --crop <Input Image Directory> <Output Image>```


Note: 
* Input and output images should be .jpg file
* Depending on your local machine, you may need to change ```python``` to ```python3```.

## Exmaple Results
Basic Image Stitching
![Basic Image Stitching](Test_Outputs/test2.jpg)

Filling Black Pixels
![Filling Black Pixels](Test_Outputs/filled_test2.jpg)

Auto-Cropping
![Auto-Cropping](Test_Outputs/coppred_test2.jpg)

## Authors
* Saem Jeon
* Abigail Clune

## References
[Original Cropping Implementation](https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/)

[Automatic Panoramic Image Stitching](http://matthewalunbrown.com/papers/ijcv2007.pdf)