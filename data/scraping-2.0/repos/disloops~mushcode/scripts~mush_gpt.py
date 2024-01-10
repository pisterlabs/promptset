# MIT License
# Copyright (c) 2023 Matt Westfall (@disloops)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = 'Matt Westfall'
__version__ = '0.2'
__email__ = 'disloops@gmail.com'

# This script functions as a localhost server that a MUSH can use to easily
# interact with the OpenAI API.

from flask import Flask, request, jsonify
from openai import AsyncOpenAI, OpenAI
import time
import os

app = Flask(__name__)

# OpenAI API key
client = OpenAI(api_key="[API key here]")

# Pseudo-secret password value that must be present in incoming requests
auth_key = "[auth value here]"

# Valid MUSH characters for which a system pre-prompt exists
prompts = ["oracle"]

# Add message buffer
message_buffer = []

@app.route('/', methods=['POST'])
def index():
    try:

        # Check for POST parameters
        json_data = request.get_json()
        if not json_data or 'text' not in json_data or 'auth' not in json_data or 'char' not in json_data:
            return jsonify({"message": "Failed: Invalid request data"}), 400

        text = json_data['text'].strip()
        auth = json_data['auth'].strip()
        char = json_data['char'].strip()

        if len(auth) > 100 or auth != auth_key:
            return jsonify({"message": "Failed: Unauthorized"}), 401

        if len(text) > 500:
            return jsonify({"message": "Failed: Input exceeds maximum length"}), 400

        if len(char) > 100 or char not in prompts:
            return jsonify({"message": "Failed: Invalid character reference"}), 400

        # Add user message to the buffer
        add_message("user", text, char)

        # Filter messages for the current character
        char_messages = get_char_messages(char)

        # Format request to OpenAI API endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            temperature=1,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1,
            messages=[
                {"role": "system", "content": prompt(char)},
                *char_messages
            ]
        )

        # Return response to caller
        finish_reason = response.choices[0].finish_reason

        if finish_reason == 'content_filter':
            return jsonify({"message": "Failed: Content filter activated"})
        elif finish_reason == 'length':
            return jsonify({"message": "Failed: Output exceeds maximum length"})
        else:
            message_content = response.choices[0].message.content
            add_message("assistant", message_content, char)
            return jsonify({"message": message_content})

    except Exception as e:
        return jsonify({"message": "Failed: Error"}), 500

# Add a message to the buffer
def add_message(role, content, character):
    global message_buffer

    message_buffer.append({"role": role, "content": content, "timestamp": time.time(), "character": character})

    # Keep only the last 5 messages
    message_buffer = message_buffer[-9:]

    # Purge messages older than 1 hour
    current_time = time.time()
    message_buffer = [msg for msg in message_buffer if current_time - msg["timestamp"] < 3600]

# Get filtered messages for the current character
def get_char_messages(character):
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in message_buffer
        if msg.get("character") == character
    ]

# Return an existing character prompt
def prompt(char):
    if char == "oracle":
        return """[Put your oracle prompt here]"""

# Use default port 5000
if __name__ == '__main__':
    app.run()