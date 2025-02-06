from flask import Blueprint, jsonify, request
from airport_configs import AIRPORTS

bp = Blueprint('routes', __name__)


@bp.route('/airports', methods=['GET'])
def get_airports():
    return jsonify(AIRPORTS)


@bp.route('/airports/<airport_code>/locations', methods=['GET'])
def get_airport_locations(airport_code):
    """Get available locations for a specific airport"""
    airport_code = airport_code.upper()

    if airport_code not in AIRPORTS:
        return jsonify({
            'error': f'Airport with code {airport_code} not found'
        }), 404

    airport = AIRPORTS[airport_code]

    return jsonify({
        'code': airport_code,
        'name': airport['name'],
        'locations': list(airport['locations'].keys())
    })


@bp.route('/airports/<airport_code>/path', methods=['GET'])
def get_path(airport_code):
    """Get path between two locations at a specific airport"""
    airport_code = airport_code.upper()
    start = request.args.get('start')
    end = request.args.get('end')

    if not start or not end:
        return jsonify({
            'error': 'Start and end locations are required'
        }), 400

    if airport_code not in AIRPORTS:
        return jsonify({
            'error': f'Airport with code {airport_code} not found'
        }), 404

    airport = AIRPORTS[airport_code]
    locations = airport['locations']

    if start not in locations or end not in locations:
        return jsonify({
            'error': 'Invalid start or end location'
        }), 400

    return jsonify({
        'start': start,
        'end': end,
        'airport': airport_code
    })
