from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import requests


def url_to_img(url):
  img = Image.open(BytesIO(requests.get(url).content))
  img.save('images/sample1.jpg')
  return None


def find_white_background(imgpath, threshold=0.3):
    url_to_img(imgpath)
    imgArr = cv2.imread('images/sample1.jpg')
    background = np.array([255, 255, 255])
    percent = (imgArr == background).sum() / imgArr.size
    if percent >= threshold:
        print(percent)
        return True
    else:
        return False
