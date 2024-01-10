import os
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

global requests_list
requests_list = []

@app.route("/")
def home():
    return jsonify(requests_list)

@app.route("/api/completions", methods=["POST"])
def completions():
    payload = request.json
    requests_list.append(payload)
    response = openai.Completion.create(
        model=payload["model"],
        prompt=payload["prompt"],
        max_tokens=payload.get("max_tokens", 100),
        temperature=payload.get("temperature", 0.8),
    )
    return jsonify(response.choices[0].text.strip())

@app.route("/api/chat/completions", methods=["POST"])
def chat():
    payload = request.json
    requests_list.append(payload)
    response = openai.ChatCompletion.create(
        model=payload["model"],
        messages=payload["messages"]
    )
    return jsonify(response)

@app.route("/api/edits", methods=["POST"])
def edits():
    payload = request.json
    requests_list.append(payload)
    response = openai.Edit.create(
        model=payload["model"],
        input=payload["input"],
        instruction=payload["instruction"]
    )
    return jsonify(response.choices[0].text)

@app.route('/api/audio/transcriptions', methods=['POST'])
def transcriptions():
    audio_file = request.files['file']
    request_info = {
        'filename': audio_file.filename,
        'content_type': audio_file.content_type,
        'headers': dict(request.headers)
    }
    requests_list.append(request_info)
    
    # save to a temporary file
    tmp_file_name = 'temp_audio.wav'
    audio_file.save(tmp_file_name)

    # use the saved file for transcription
    with open(tmp_file_name, 'rb') as f:
        response = openai.Audio.transcribe("whisper-1", f)

    # delete the temporary file
    os.remove(tmp_file_name)
    
    return jsonify(response)

@app.route("/api/images", methods=["POST"])
def images():
    payload = request.json
    requests_list.append(payload)
    response = openai.Image.create(
        prompt=payload["prompt"],
        n=payload.get("n", 1),
        size=payload.get("size", "512x512")
    )
    return jsonify(response)

@app.route("/api/images/edits", methods=["POST"])
def image_edits():
    image_file = request.files["image"]
    prompt = request.form["prompt"]
    n = int(request.form.get("n", 1))
    size = request.form.get("size", "1024x1024")
    response_format = request.form.get("response_format", "url")

    request_info = {
        'filename': image_file.filename,
        'content_type': image_file.content_type,
        'headers': dict(request.headers),
        'prompt': prompt,
        'n': n,
        'size': size,
        'response_format': response_format
    }
    requests_list.append(request_info)

    response = openai.Image.create_edit(
        image=image_file,
        prompt=prompt,
        n=n,
        size=size,
        response_format=response_format
    )

    return jsonify(response)

@app.route("/api/images/variations", methods=["POST"])
def image_variations():
    image_file = request.files["image"]
    n = request.form.get("n", 1)
    size = request.form.get("size", "1024x1024")

    request_info = {
        'filename': image_file.filename,
        'content_type': image_file.content_type,
        'headers': dict(request.headers),
        'n': n,
        'size': size
    }
    requests_list.append(request_info)

    response = openai.Image.create_variation(
        image=image_file,
        n=int(n),
        size=size
    )
    return jsonify(response)

@app.route("/api/moderations", methods=["POST"])
def moderations():
    payload = request.json
    requests_list.append(payload)
    response = openai.Moderation.create(
        input=payload["input"]
    )
    return jsonify(response.results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
