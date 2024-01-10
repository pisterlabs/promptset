from flask import Blueprint, request, jsonify
import openai
from utilities.transcript import transcript
from dotenv import load_dotenv
import os

load_dotenv()

transcript_controller = Blueprint("transcript_controller", __name__)

openai_api_key = os.getenv("OPENAI_KEY")

# Initialize the OpenAI API client
openai.api_key = openai_api_key

@transcript_controller.route('/getTranscript', methods=['POST'])
def get_transcript():
    try:
        data = request.get_json()
        videoURL = data.get('videoURL')

        if videoURL is None:
            return jsonify({"error": "Missing 'videoURL' parameter"}), 400

        toTranscript = transcript(videoURL)

        user_message = "Add proper punctuations to the following piece of text and split it into multiple paragraphs for better readability: " + toTranscript

        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}],
        )

        response_text = chat_completion['choices'][0]['message']['content']

        return jsonify({"openaiAPIResult": response_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
