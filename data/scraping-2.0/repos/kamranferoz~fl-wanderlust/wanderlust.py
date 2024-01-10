from flask import Flask, render_template, request, jsonify
import os
from openai import NotFoundError, OpenAI

app = Flask(__name__)

# Flask doesn't natively support reactive programming as in solara. For this purpose, you'll need to maintain state on client-side or use sessions/cookies for server-side
center_default = (0, 0)
zoom_default = 2
messages = []
zoom_level = zoom_default
center = center_default
markers = []

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4-1106-preview"

def update_map(longitude, latitude, zoom):
    center = (latitude, longitude)
    zoom_level = zoom
    return "Map updated"

def add_marker(longitude, latitude, label):
    markers.append({"location": (latitude, longitude), "label": label})
    return "Marker added"

functions = {
    "update_map": update_map,
    "add_marker": add_marker,
}

@app.route('/map/update', methods=['POST'])
def map_update():
    body = request.json
    message = functions['update_map'](body['longitude'], body['latitude'], body['zoom'])
    return jsonify({'message': message})

@app.route('/marker/add', methods=['POST'])
def marker_add():
    body = request.json
    message = functions['add_marker'](body['longitude'], body['latitude'], body['label'])
    return jsonify({'message': message})

if __name__ == "__main__":
    app.run(debug=True)
