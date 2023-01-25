from flask import request, Blueprint
from .helpers import (
    find_white_background
)


r_blueprint = Blueprint('route_blueprint','__name__')


@r_blueprint.route('/',  methods=['POST'])
def check_image():
    images = request.get_json()
    is_white_bg = False
    results = []
    for image in images['images']:
        if find_white_background(image):
            is_white_bg = True
        results.append(
            {
                'Image' : image,
                'White Background':is_white_bg
            }
        )
    return results
