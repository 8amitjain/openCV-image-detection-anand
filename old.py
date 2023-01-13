import cv2
import numpy as np
from PIL import Image, ImageChops


# TODO input as link

# TODO Create this as an api
# TODO Deploy on aws
# Each funcation return a true or false the value of true or false is diffrent eg


def is_white_background(img_path, threshold=0.2):
    """remove images with transparent or white background"""
    img_arr = cv2.imread(img_path)
    background = np.array([255, 255, 255])
    percent = (img_arr == background).sum() / img_arr.size
    print(percent)
    if percent >= threshold:
        return True
    else:
        return False


def is_image_size_greater_than_1000px(img_path, height=1000, width=1000):
    img_arr = cv2.imread(img_path)

    # using shape property to get the dimensions of the image
    dimensions = img_arr.shape
    image_height = dimensions[0]
    image_width = dimensions[1]
    print(dimensions)
    if image_height >= height and image_width >= width:
        return True
    return False


def is_greyscale(img_path):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    img_arr = cv2.imread(img_path)
    image = Image.open(img_arr).convert('RGB')
    if image.mode not in ("L", "RGB"):
        raise ValueError("Unsupported image mode")

    if image.mode == "RGB":
        rgb = image.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


def is_aspect_ratio_1(img_path):
    img_arr = cv2.imread(img_path)

    # using shape property to get the dimensions of the image
    dimensions = img_arr.shape
    image_height = dimensions[0]
    image_width = dimensions[1]

    aspect_ratio = float(image_width) / image_height
    # print(aspect_ratio)
    if aspect_ratio == 1:
        return True
    return False


def is_blurry(img_path):
    img_arr = cv2.imread(img_path)
    img = cv2.imread(img_arr, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    print(laplacian_var)

    if laplacian_var > 500:
        return True
    return False


# TODO Detects as different image for same image with different orientation
# TODO update for set of images input using a best time complexity possible
#  help from https://www.youtube.com/watch?v=7INNmrmFz3Q
def is_repeated_in_set(img_path, img_path_2):
    img_arr = cv2.imread(img_path)
    img_arr_2 = cv2.imread(img_path_2)

    return img_arr.shape == img_arr_2.shape and not(np.bitwise_xor(img_arr, img_arr_2).any())

    # Not working with accuracy
    # if original.shape == duplicate.shape:
    #     difference = cv2.subtract(original, duplicate)
    #     b, g, r = cv2.split(difference)
    #     print(cv2.countNonZero(b))
    #
    #     if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    #         return True
    #     return False
    # return False


# TODO print(is_human_in_image("Images/13.jpg")) does not dected as face
def is_human_in_image(img_path):
    image = cv2.imread(img_path)

    face_cascade = cv2.CascadeClassifier("Models/face_detect_model.xml")  # Load the cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert into grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces
    # print(faces, "no faces")

    if len(faces) > 0:  # If face are detected skip rest execution
        return True

    hog = cv2.HOGDescriptor()  # initialize the HOG descriptor
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (humans, _) = hog.detectMultiScale(image, winStride=(10, 10), padding=(32, 32), scale=1.1)  # detect humans in image
    # print('Human Detected : ', len(humans))   # getting no. of human detected

    if len(humans) > 0:
        return True
    return False


def is_product_cropped(img_path):
    img_arr = cv2.imread(img_path)

    image_rotated = cv2.rotate(img_arr, cv2.ROTATE_90_CLOCKWISE)

    def iterate_line(img, r_thresh, g_thresh, b_thresh, y):
        """
            This function returns true only when a given row at a height "y"
            from the origin(top - left) if fully black and false othrwise
        """
        for i in img[y]:
            if not ((i[0] > r_thresh) and (i[1] > g_thresh) and i[2] > b_thresh):
                return False
        return True

    def has_border(image):
        height, width, channels = image.shape

        border_threshold_R = 250
        border_threshold_G = 250
        border_threshold_B = 250

        top_border_height = 0
        bottom_border_height = 0
        corrected_top_border_height = 0
        corrected_bottom_border_height = 0

        for i in range(height // 2):
            mid_pixel_top_half = image[i][width // 2]
            R, G, B = mid_pixel_top_half[2], mid_pixel_top_half[1], mid_pixel_top_half[0]
            if (R > border_threshold_R) and (G > border_threshold_G) and (B > border_threshold_B):
                top_border_height += 1
            else:
                break

        for i in range(height - 1, (height // 2) - 1, -1):
            mid_pixel_bottom_half = image[i][width // 2]
            R, G, B = mid_pixel_bottom_half[2], mid_pixel_bottom_half[1], mid_pixel_bottom_half[0]
            if (R > border_threshold_R) and (G > border_threshold_G) and (B > border_threshold_B):
                bottom_border_height += 1
            else:
                break

        if (top_border_height > 1) and (bottom_border_height > 1):

            for i in range(top_border_height):
                if iterate_line(image, border_threshold_R, border_threshold_G, border_threshold_B, i):
                    corrected_top_border_height += 1
                else:
                    break

            for i in range(height - 1, height - 1 - bottom_border_height, -1):
                if iterate_line(image, border_threshold_R, border_threshold_G, border_threshold_B, i):
                    corrected_bottom_border_height += 1
                else:
                    break
            # print(corrected_top_border_height, corrected_bottom_border_height, "XR")

            if corrected_bottom_border_height > 1 and corrected_top_border_height > 1:
                return True
            else:
                return False
        else:
            return False

    if has_border(img_arr) and has_border(image_rotated):
        return True
    return False


def is_sketch(img_path):
    img_arr = cv2.imread(img_path)

    output = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    # Apply gaussian blur
    output = cv2.GaussianBlur(output, (3, 3), 0)

    # detect edges in the image
    output = cv2.Laplacian(output, -1, ksize=5)

    # invert the binary image
    output = 255 - output

    # binary thresholding
    ret, output = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("image", img_arr)
    cv2.imshow("pencilsketch", output)
    cv2.waitKey(0)

    return img_arr.shape == output.shape and not(np.bitwise_xor(img_arr, output).any())

    # create widnows to dispplay images
    # cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("pencilsketch", cv2.WINDOW_AUTOSIZE)


def is_product(img_path):
    img_arr = cv2.imread(img_path)

    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Two pass dilate with horizontal and vertical kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)

    # Find contours, filter using contour threshold area, and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 20000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (36, 255, 12), 3)

    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilate)
    cv2.imshow('image', img_arr)
    cv2.waitKey()


# print(is_repeated_in_set("Images/9.jpeg", "Images/9 copy.jpeg"))
# print(is_repeated_in_set("Images/5.png", "Images/5 copy.png"))
# print(is_repeated_in_set("Images/10.png", "Images/10.png"))


# print(is_white_background("Images/1.jpg"))
# print(is_sketch("Images/20.jpeg"))



# is_product("Images/7.jpg")
