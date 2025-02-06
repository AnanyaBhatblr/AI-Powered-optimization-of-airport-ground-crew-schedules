from flask import Blueprint, jsonify
from airport_configs import AIRPORTS

bp = Blueprint('airport_routes', __name__)


@bp.route('/airports', methods=['GET'])
def get_airports():
    return jsonify(AIRPORTS)
