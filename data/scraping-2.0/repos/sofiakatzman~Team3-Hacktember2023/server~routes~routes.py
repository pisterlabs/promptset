from .user import *
from .auth import *
from .content import *
from config import OPENAI_API_KEY, requests, app, request, os
from flask_cors import CORS, cross_origin

# chat bot route
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
@cross_origin(origin='https://hackathonsubmission.onrender.com', headers=['Content-Type'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200

    payload = request.json
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    return jsonify(response.json())
