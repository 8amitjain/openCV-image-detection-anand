from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import requests


def find_image_format(img):
    return img.format


def url_to_img(url):
  img = Image.open(BytesIO(requests.get(url).content))
  format = find_image_format(img)
  img.save(f'images/sample.{format}')
  return format


def find_white_background(imgpath, threshold=0.3):
    f = url_to_img(imgpath)
    imgArr = cv2.imread(f'images/sample.{f}')
    background = np.array([255, 255, 255])
    percent = (imgArr == background).sum() / imgArr.size
    if percent >= threshold:
        return True
    else:
        return False
