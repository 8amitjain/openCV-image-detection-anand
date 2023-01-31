from flask import request, Blueprint, jsonify
from .helpers import (
    find_white_background,
    is_human_in_image,
    is_greyscale,
    is_blurry,
    is_aspect_ratio_1,
    is_image_size_greater_than_1000px
)


r_blueprint = Blueprint('route_blueprint','__name__')


@r_blueprint.route('/',  methods=['POST'])
def check_image():
    images = request.get_json()
    results = {'status':'200','message':'success','data':[]}
    for image in images['images']:
        results['data'].append(
                {
                    'Image URL' : image, # Return image url
                    'White Background' : True if find_white_background(image) else False, # Return True if image has white background else False.
                    'Human in Image' : True if is_human_in_image(image) else False, # Return True if human in image else False.
                    'Greyscale Image' : True if is_greyscale(image) else False, # Return True if image is greyscale else False.
                    'Blurry Image' : True if is_blurry(image) else False, # Return True if image is blurry else False.
                    'Aspect Ratio 1' : True if is_aspect_ratio_1(image) else False, # Return True if image is aspect ratio is 1 else False.
                    'Greater than 1000px' : True if is_image_size_greater_than_1000px(image) else False, # Return True if image is greater than else False.
                }
        )
    return jsonify(results)
