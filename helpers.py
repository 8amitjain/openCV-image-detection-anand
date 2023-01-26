from PIL import Image, ImageChops
from io import BytesIO
import cv2
import numpy as np
import requests
import urllib


'''
This function find an image format like png, jpg etc.
'''
def find_image_format(img):
    return img.format


'''
This function convert image to url and save it to images directory.
'''
def url_to_img(url,file_name):
  img = Image.open(BytesIO(requests.get(url).content))
  format = find_image_format(img)
  img.save(f'images/{file_name}.{format}')
  return format


'''
This function will find any white background in images.
'''
def find_white_background(imgpath, threshold=0.3):
    file_name = 'white_background_image'
    f = url_to_img(url=imgpath,file_name=file_name)
    imgArr = cv2.imread(f'images/{file_name}.{f}')
    background = np.array([255, 255, 255])
    percent = (imgArr == background).sum() / imgArr.size
    if percent >= threshold:
        return True
    else:
        return False


'''
This function will find human in image.
'''
def is_human_in_image(imgpath):
    file_name='human_in_image'
    f = url_to_img(url=imgpath,file_name=file_name)
    img_path = f'Images/{file_name}.{f}'
    image = cv2.imread(img_path)

    face_cascade = cv2.CascadeClassifier("Models/face_detect_model.xml")  # Load the cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert into grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces

    if len(faces) > 0:  # If face are detected skip rest execution
        return True

    hog = cv2.HOGDescriptor()  # initialize the HOG descriptor
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (humans, _) = hog.detectMultiScale(image, winStride=(10, 10), padding=(32, 32), scale=1.1)  # detect humans in image
    if len(humans) > 0:
        return True
    return False
