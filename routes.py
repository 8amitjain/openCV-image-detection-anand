from flask import request, Blueprint, jsonify
from .helpers import (
    find_white_background
)


r_blueprint = Blueprint('route_blueprint','__name__')


@r_blueprint.route('/',  methods=['POST'])
def check_image():
    images = request.get_json()
    results = {'status':'200','message':'success','data':[]}
    for image in images['images']:
        results['data'].append(
                {
                    'image_url' : image,
                    'White Background': True if find_white_background(image) else False
                }
        )
    return jsonify(results)
