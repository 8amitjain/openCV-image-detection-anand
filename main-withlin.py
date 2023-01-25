from PIL import Image, ImageChops
from sentence_transformers import SentenceTransformer, util
from io import BytesIO
from flask import Flask, request
from collections import Counter
import cv2
import numpy as np
import requests
import urllib
from io import BytesIO
import requests
import os

from .routes import r_blueprint

# from StringIO import StringIO

# todo check if the bg of image is white.
# todo Find objects apart from white color & check their size in reference to image
# todo see if the biggest object is touching the border
# check the size of biggest object is 85% of the image or not


app = Flask(__name__)
app.register_blueprint(r_blueprint)
"""Each function return a true or false the value of true or false is different eg"""
# TODO fix the issue with images which has a logo on one side.
# TODO handle other images types png & gif
# TODO Deploy on aws

# Later
# print(is_human_in_image("Images/13.jpg")) does not detected as face / optimization needed


def read_image_from_url(link):
    # req = urllib.request.urlopen(link)
    # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # img_arr = cv2.imdecode(arr, -1)  # 'Load it as it is'
    # return img_arr
    url = link['images']
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def is_white_background(img_arr):
        img = img_arr
        manual_count = {}
        w, h, channels = img.shape
        total_pixels = w * h
        number_counter = 0

        def count():
            for y in range(0, h):
                for x in range(0, w):
                    RGB = (img[x, y, 2], img[x, y, 1], img[x, y, 0])
                    if RGB in manual_count:
                        manual_count[RGB] += 1
                    else:
                        manual_count[RGB] = 1

        def average_colour():
            red = 0
            green = 0
            blue = 0
            sample = 10
            for top in range(0, sample):
                red += number_counter[top][0][0]
                green += number_counter[top][0][1]
                blue += number_counter[top][0][2]

            average_red = red / sample
            average_green = green / sample
            average_blue = blue / sample
            # print("Average RGB for top ten is: (", average_red,
            #       ", ", average_green, ", ", average_blue, ")")

        def twenty_most_common():
            count()
            number_counter = Counter(manual_count).most_common(20)
            return number_counter
            # for rgb, value in number_counter:
            #     print(rgb, value, ((float(value) / total_pixels) * 100))

        def detect():
            numbered_counter = twenty_most_common()
            percentage_of_first = (
                    float(numbered_counter[0][1]) / total_pixels)
            print(percentage_of_first)
            print("Background color is ", numbered_counter[0][0])
            if percentage_of_first >= 0.2:
                return True
            else:
                return False
            # if percentage_of_first > 0.5:
            # else:
            #     average_colour()

        detect()


# 96% accuracy on volition images
# 50% accuracy on good images
# def is_white_background(img_arr, threshold=0.2):
#     """remove images with transparent or white background"""
#     background = np.array([255, 255, 255])
#     percent = (img_arr == background).sum() / img_arr.size
#     print(percent)
#     # cv2.imshow("img_arr", img_arr)
#     # cv2.waitKey(0)
#     if percent >= threshold:
#         return True
#     else:
#         return False


def is_image_size_greater_than_1000px(img_arr, height=1000, width=1000):
    # using shape property to get the dimensions of the image
    dimensions = img_arr.shape
    image_height = dimensions[0]
    image_width = dimensions[1]
    # print(dimensions)
    if image_height >= height and image_width >= width:
        return True
    return False


# 99% accuracy on both (violated data is 90% wrong)
def is_greyscale(img_link):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """

    urllib.request.urlretrieve(img_link, 'Images/is_greyscale.jpg')  # Save the image for use as
    image = Image.open('Images/is_greyscale.jpg').convert('RGB')

    if image.mode not in ("L", "RGB"):
        raise ValueError("Unsupported image mode")

    if image.mode == "RGB":
        rgb = image.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


def is_aspect_ratio_1(img_arr):
    # using shape property to get the dimensions of the image
    dimensions = img_arr.shape
    image_height = dimensions[0]
    image_width = dimensions[1]

    aspect_ratio = float(image_width) / image_height
    # print(aspect_ratio)
    if aspect_ratio == 1:
        return True
    return False


def is_blurry(img_link):
    urllib.request.urlretrieve(img_link, 'Images/is_blurry.jpg')  # Save the image for use as
    img = cv2.imread('Images/is_blurry.jpg', cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    print(laplacian_var, "LV")

    if laplacian_var < 200:
        return True
    return False


def is_repeated_in_set(images_list):
    # model = SentenceTransformer('clip-ViT-B-32')  # Load the OpenAI CLIP Model
    model = SentenceTransformer('clip-ViT-L-14')  # Load the OpenAI CLIP Model

    # image_names = list(glob.glob('./*.jpg'))  # Compute the embeddings

    def link_to_image(img_link):
        response = requests.get(img_link)
        return response.content

    encoded_image = model.encode([Image.open(BytesIO(link_to_image(filepath))) for filepath in images_list],
                                 batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    """Now we run the clustering algorithm. This function compares images against all other images and returns a list with 
     the pairs that have the highest cosine similarity score"""

    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    NUM_SIMILAR_IMAGES = 10

    # """ Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and is scored in decreasing order
    # A duplicate image will have a score of 1.00 """
    duplicates = [image for image in processed_images if image[0] >= 1]
    similar_images = []

    # # Output the top X duplicate images
    for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
        print("\nScore: {:.3f}%".format(score * 100))
        print(images_list[image_id1])
        print(images_list[image_id2])
        similar_images.append([images_list[image_id1], images_list[image_id2], score * 100])

    # NEAR DUPLICATES
    """ Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
    you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
    A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
    duplicate images, we can set it at 0.99 or any number 0 < X < 1.00."""

    # threshold = 0.90
    # near_duplicates = [image for image in processed_images if image[0] > threshold]
    #
    # similar_images = []
    # for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
    #     print("\nScore: {:.3f}%".format(score * 100))
    #     print(images_list[image_id1])
    #     print(images_list[image_id2])
    #     similar_images.append([images_list[image_id1], images_list[image_id2], score * 100])
    return similar_images
# can use hash of image to check for duplicate or
# https://stackoverflow.com/questions/3383892/is-it-possible-to-detect-duplicate-image-files
# or work on below code
# def is_repeated_in_set(images_list):
#     def check_image_is_same(img_path, img_path_2):
#         # img_arr = read_image_from_url(img_path)
#         # img_arr_2 = read_image_from_url(img_path_2)
#
#         img_arr = cv2.imread(img_path)
#         img_arr_2 = cv2.imread(img_path_2)
#
#         # return img_arr.shape == img_arr_2.shape and not (np.bitwise_xor(img_arr, img_arr_2).any())
#
#         def check_rotation_in_angle(img_arr, img_arr_2):
#             image_arr_rotated = img_arr
#             for _ in range(4):
#                 image_arr_rotated = cv2.rotate(image_arr_rotated, cv2.ROTATE_90_CLOCKWISE)
#                 # cv2.imshow("image_arr_rotated", image_arr_rotated)
#                 # cv2.waitKey()
#                 #
#                 # cv2.imshow("img_arr_2", img_arr_2)
#                 # cv2.waitKey()
#
#                 print(np.bitwise_xor(image_arr_rotated, img_arr_2), "NOR")
#                 if image_arr_rotated.shape == img_arr_2.shape and not (np.bitwise_xor(image_arr_rotated, img_arr_2).any()):
#                     print("in", "RN")
#                     return True
#
#         if check_rotation_in_angle(img_arr, img_arr_2):
#             return True
#
#         if check_rotation_in_angle(img_arr_2, img_arr):
#             return True
#         return False
#
#     duplicated_images = []
#     for index in range(len(images_list)):
#         for index_2 in range(len(images_list)):
#             if index != index_2 and check_image_is_same(images_list[index], images_list[index_2]):
#                 duplicated_images.append([images_list[index], images_list[index_2], index, index_2])
#     return duplicated_images


# 95% Accuracy - their AI detect humans part as well as human as hand or legs
def is_human_in_image(img_link):
    urllib.request.urlretrieve(img_link, 'Images/is_human_in_image.jpg')  # Save the image for use as
    img_path = 'Images/is_human_in_image.jpg'
    image = cv2.imread(img_path)

    face_cascade = cv2.CascadeClassifier("Models/face_detect_model.xml")  # Load the cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert into grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces
    # print(faces, "no faces")

    if len(faces) > 0:  # If face are detected skip rest execution
        # print("face", len(faces))
        return True

    hog = cv2.HOGDescriptor()  # initialize the HOG descriptor
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (humans, _) = hog.detectMultiScale(image, winStride=(10, 10), padding=(32, 32), scale=1.1)  # detect humans in image
    # print('Human Detected : ', len(humans))   # getting no. of human detected
    if len(humans) > 0:
        return True
    return False


# def is_product_cropped(img_arr):
#     image_rotated = cv2.rotate(img_arr, cv2.ROTATE_90_CLOCKWISE)
#
#     def iterate_line(img, r_thresh, g_thresh, b_thresh, y):
#         """
#             This function returns true only when a given row at a height "y"
#             from the origin(top - left) if fully black and false othrwise
#         """
#         for i in img[y]:
#
#             img[y] = [255, 0, 0]
#             if not ((i[0] > r_thresh) and (i[1] > g_thresh) and i[2] > b_thresh):
#                 cv2.imshow("C", img)
#                 cv2.waitKey()
#                 return False
#         cv2.imshow("D", img)
#         cv2.waitKey()
#         return True
#
#     def has_border(image):
#         height, width, channels = image.shape
#
#         border_threshold_R = 250
#         border_threshold_G = 250
#         border_threshold_B = 250
#
#         top_border_height = 0
#         bottom_border_height = 0
#         corrected_top_border_height = 0
#         corrected_bottom_border_height = 0
#
#         for i in range(height // 2):
#             mid_pixel_top_half = image[i][width // 2]
#             image = cv2.circle(image, (i, width // 2), radius=4, color=(0, 0, 255), thickness=-1)
#             image = cv2.circle(image, (i, height // 2), radius=4, color=(255, 255, 0), thickness=-1)
#
#             R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
#             if (R > border_threshold_R) and (G > border_threshold_G) and (B > border_threshold_B):
#                 top_border_height += 1
#             else:
#                 break
#         cv2.imshow("A", image)
#         cv2.waitKey()
#         for i in range(height - 1, (height // 2) - 1, -1):
#             mid_pixel_bottom_half = image[i][width // 2]
#             image = cv2.circle(image, (i, width // 2), radius=4, color=(0, 255, 0), thickness=-1)
#             image = cv2.circle(image, (i, height // 2), radius=4, color=(255, 0, 0), thickness=-1)
#
#             R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
#             if (R > border_threshold_R) and (G > border_threshold_G) and (B > border_threshold_B):
#                 bottom_border_height += 1
#             else:
#                 break
#         cv2.imshow("B", image)
#         cv2.waitKey()
#         if (top_border_height > 1) and (bottom_border_height > 1):
#
#             for i in range(top_border_height):
#                 if iterate_line(image, border_threshold_R, border_threshold_G, border_threshold_B, i):
#                     corrected_top_border_height += 1
#                 else:
#                     break
#
#             for i in range(height - 1, height - 1 - bottom_border_height, -1):
#                 if iterate_line(image, border_threshold_R, border_threshold_G, border_threshold_B, i):
#                     corrected_bottom_border_height += 1
#                 else:
#                     break
#
#             print(top_border_height, bottom_border_height)
#             print(corrected_top_border_height, corrected_bottom_border_height, "XR")
#
#             if corrected_bottom_border_height > 1 and corrected_top_border_height > 1:
#                 return True
#             else:
#                 return False
#         else:
#             return False
#
#     # has_border(img_arr)
#     # has_border(image_rotated)
#
#     if has_border(img_arr) and has_border(image_rotated):
#         return True
#     return False


# def is_sketch(img_arr):
#     output = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
#
#     # Apply gaussian blur
#     output = cv2.GaussianBlur(output, (3, 3), 0)
#
#     # detect edges in the image
#     output = cv2.Laplacian(output, -1, ksize=5)
#
#     # invert the binary image
#     output = 255 - output
#
#     # binary thresholding
#     ret, output = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY)
#     cv2.imshow("image", img_arr)
#     cv2.imshow("pencilsketch", output)
#     cv2.waitKey(0)
#
#     return img_arr.shape == output.shape and not(np.bitwise_xor(img_arr, output).any())
#
#     # create widnows to dispplay images
#     # cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
#     # cv2.namedWindow("pencilsketch", cv2.WINDOW_AUTOSIZE)

#
# def is_product(img_arr):
#     # Load image, grayscale, Gaussian blur, Otsu's threshold
#     gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (13, 13), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
#     # Two pass dilate with horizontal and vertical kernel
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
#     dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
#     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
#     dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)
#
#     # Find contours, filter using contour threshold area, and draw rectangle
#     cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     for c in cnts:
#         area = cv2.contourArea(c)
#         if area > 20000:
#             x, y, w, h = cv2.boundingRect(c)
#             cv2.rectangle(img_arr, (x, y), (x + w, y + h), (36, 255, 12), 3)
#
#     cv2.imshow('thresh', thresh)
#     cv2.imshow('dilate', dilate)
#     cv2.imshow('image', img_arr)
#     cv2.waitKey()


# print(is_repeated_in_set("Images/9.jpeg", "Images/9 copy.jpeg"))
# print(is_repeated_in_set("Images/5.png", "Images/5 copy.png"))
# print(is_repeated_in_set("Images/10.png", "Images/10.png"))


# print(is_white_background("Images/1.jpg"))
# print(is_white_background("https://answers.opencv.org/upfiles/logo_2.png"))
# print(is_sketch("Images/20.jpeg"))


# print(is_greyscale("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/71+WeWkajOL._SL1200_.jpg"))
# print(is_greyscale("https://i.stack.imgur.com/FvYYq.jpg"))
# print(is_greyscale("https://i0.wp.com/digital-photography-school.com/wp-content/uploads/2017/06/dps-why-choose-greyscale-004.jpg"))
# print(is_greyscale("https://i.stack.imgur.com/jECAG.png"))

# print(is_human_in_image("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/611QYEP7ubL._SL1500_.jpg"))
# print(is_human_in_image("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/21hcvFUEtWL.jpg"))
# print(is_human_in_image("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/510lOSXIqQL.jpg"))
# print(is_human_in_image("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/61tRRPOnQLL._AC_UL640_FMwebp_QL65_.jpg"))

# images = ["https://m.media-amazon.com/images/I/51wfv+ajcwL._SX569_.jpg",
#           "https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/61woEr3q-GS._AC_UL640_FMwebp_QL65_.jpg",
#           "https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/61woEr3q-GS._AC_UL640_FMwebp_QL65_.jpg",
#           "https://m.media-amazon.com/images/I/61EYcOFtXAL._AC_UL640_QL65_.jpg"]

# print(is_repeated_in_set(images))

# 50% accuracy on correct data mostly detecting wrong images with 1 px space.
# 70% accuracy on Incorrect data mostly detecting images with no white bg.
def is_entire_product_visible(img_link):
    urllib.request.urlretrieve(img_link, 'Images/is_human_in_image.jpg')  # Save the image for use as
    img_path = 'Images/is_human_in_image.jpg'
    image = cv2.imread(img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)

    # find the contours in the edged image
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    # print(len(contours), "objects were found in this image.")

    # get bounds of white pixels
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    white = np.where(thresh == 255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    # print(xmin, xmax, ymin, ymax)

    # crop the gray image at the bounds
    crop = gray[ymin:ymax, xmin:xmax]
    hh, ww = crop.shape

    # get contours (presumably just one) and its bounding box
    # contours = contours[0] if len(contours) == 2 else contours[1]
    touches_border = []
    area_list = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)

        # draw bounding box on input
        # cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # cv2.imshow("image_copy", image_copy)
        # cv2.waitKey(0)
        # test if contour touches sides of image

        area = (w * h)
        area_list.append(area)
        if x == 0 or y == 0 or x + w == ww or y + h == hh:
            # print('region touches the sides')
            touches_border.append([area, True])
        else:
            # print('region does not touch the sides')
            touches_border.append([area, False])
    # print(max(area_list))

    for area in touches_border:
        if area[0] == max(area_list):
            if area[1] is True:
                return False
    return True
    # cv2.imshow("contours", image_copy)
    # cv2.waitKey(0)

    # # threshold
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #
    # # get bounds of white pixels
    # white = np.where(thresh == 255)
    # xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    # print(xmin, xmax, ymin, ymax)
    #
    # # crop the gray image at the bounds
    # crop = gray[ymin:ymax, xmin:xmax]
    # hh, ww = crop.shape
    #
    # # do adaptive thresholding
    # thresh2 = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1.1)
    #
    # # apply morphology
    # kernel = np.ones((1, 7), np.uint8)
    # morph = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((5, 5), np.uint8)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    #
    # # invert
    # morph = 255 - morph

    # get contours (presumably just one) and its bounding box
    # contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours), "contours")
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # for cntr in contours:
    #     x, y, w, h = cv2.boundingRect(cntr)
    #     # cv2.putText(image, str(w), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 1)
    #
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    #
    # # draw bounding box on input
    # bbox = image.copy()
    # cv2.rectangle(bbox, (x+xmin, y+ymin), (x+xmin+w, y+ymin+h), (0, 0, 255), 1)
    #
    # # test if contour touches sides of image
    # if x == 0 or y == 0 or x+w == ww or y+h == hh:
    #     print('region touches the sides')
    # else:
    #     print('region does not touch the sides')
    #
    # # # save resulting masked image
    # # cv2.imwrite('streak_thresh.png', thresh)
    # # cv2.imwrite('streak_crop.png', crop)
    # cv2.imshow('contours', bbox)
    #
    # # # display result
    # # cv2.imshow("thresh", thresh)
    # # cv2.imshow("crop", crop)
    # # cv2.imshow("thresh2", thresh2)
    # # cv2.imshow("morph", morph)
    # # cv2.imshow("bbox", bbox)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# print(is_product_img_link("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/41PY+iCUi4L._SL1500_.jpg"))
# print(is_product_img_link("https://m.media-amazon.com/images/I/91yuGwfJxQL._SX679_PIbundle-3,TopRight,0,0_SX679SY/790SH20_.jpg"))
# is_product_cropped(read_image_from_url("https://m.media-amazon.com/images/I/91yuGwfJxQL._SX679_PIbundle-3,TopRight,0,0_SX679SY790SH20_.jpg"))


# is_white_background(read_image_from_url("https://m.media-amazon.com/images/I/511U11I0afL._AC._SR360,460.jpg"))
# is_white_background(read_image_from_url("https://images-eu.ssl-images-amazon.com/images/I/71R-TNQ5D-L._AC_UL900_SR900,600_.jpg"))
# is_white_background(read_image_from_url("https://images-eu.ssl-images-amazon.com/images/I/71q0r4djtKS._AC_UL900_SR900,600_.jpg"))
# is_white_background(read_image_from_url("https://m.media-amazon.com/images/I/51LgFtoOg7L._AC_UL640_QL65_.jpg"))


# is_product_img_link("https://m.media-amazon.com/images/I/91DULdQJPHL._AC_UL640_QL65_.jpg")
# is_product_img_link("https://m.media-amazon.com/images/I/51y4DrDvItL._AC_UY436_FMwebp_QL65_.jpg")
# is_product_img_link("https://m.media-amazon.com/images/I/71wlmIKWaoL._AC_UL640_FMwebp_QL65_.jpg")
# is_product_img_link("https://m.media-amazon.com/images/I/61f0qYWrxkL._AC_UY436_QL65_.jpg")

# TODO manage for other URL format
# TODO handle 403 error with URI


# def find_white_background(imgpath, threshold=0.3):
#     url_to_img(imgpath)
#     imgArr = cv2.imread('images/sample1.jpg')
#     background = np.array([255, 255, 255])
#     percent = (imgArr == background).sum() / imgArr.size
#     if percent >= threshold:
#         print(percent)
#         return True
#     else:
#         return False

# @app.route('/',  methods=['POST'])
# def check_image():
#     images = request.get_json()
#     is_white_bg = False
#     results = []
#     for image in images['images']:
#         if find_white_background(image):
#             is_white_bg = True
#         results.append(
#             {
#                 'Image' : image,
#                 'White Background':is_white_bg
#             }
#         )
#     return results


def check_images():
    images = request.get_json()
    results = []
    is_white_bg = True
    is_product_visible = True
    greyscale = False
    is_human = False
    for image in images['images']:
        if not is_white_background(read_image_from_url(image)):
            is_white_bg = False
        if not is_entire_product_visible(image):
            is_product_visible = False
        if is_greyscale(image):
            greyscale = True
        if is_human_in_image(image):
            is_human = True

    return results

if __name__ == "__main__":
    app.run(port=8000, debug=True)
# app.run()

