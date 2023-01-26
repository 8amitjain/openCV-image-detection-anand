from flask import request, Blueprint, jsonify
from .helpers import (
    find_white_background,
    is_human_in_image,
    is_greyscale
)


r_blueprint = Blueprint('route_blueprint','__name__')


@r_blueprint.route('/',  methods=['POST'])
def check_image():
    images = request.get_json()
    results = {'status':'200','message':'success','data':[]}
    for image in images['images']:
        results['data'].append(
                {
                    'Image URL' : image, # Image url
                    'White Background' : True if find_white_background(image) else False, # Return True if image has white background else False.
                    'Human in Image' : True if is_human_in_image(image) else False, # Return True if human in image else False.
                    'Greyscale Image' : True if is_greyscale(image) else False, # Return True if image is greyscale else False.
                }
        )
    return jsonify(results)
