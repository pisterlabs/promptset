from flask import Blueprint, jsonify, request
from models.itinerary import write_itinerary
from models.openai_service import OpenAIService

itinerary_blueprint = Blueprint('itinerary', __name__)

@itinerary_blueprint.route('/',methods = ["POST"])
def get_itinerary():
    openai_service = OpenAIService()
    data = request.get_json()
    city = data.get('city')
    duration = data.get('duration')
    user_id = request.headers.get('Authorization')

    if not city or not duration:
        return jsonify({'error': 'City and duration are required parameters'}), 400
    
    itinerary_data = openai_service.generate_itinerary(city, duration)

    response_data = jsonify(itinerary_data)
    write_itinerary(user_id, itinerary_data)
    print(response_data)
    return response_data