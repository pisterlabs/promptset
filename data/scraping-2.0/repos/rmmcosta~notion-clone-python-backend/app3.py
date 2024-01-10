from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
client = OpenAI()

@app.route('/api/completion', methods=['POST','OPTIONS'])
def completion():
    data = request.get_json()
    prompt = data['prompt']

    start_time = time.time()

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                'role': 'system',
                'content': '''You are a helpful AI embedded in a notion text editor app that is used to autocomplete sentences.
                The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
                AI is a well-behaved and well-mannered individual.
                AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user.'''
            },
            {
                'role': 'user',
                'content': '''I am writing a piece of text in a notion text editor app.
                Help me complete my train of thought here: ##{}##
                keep the tone of the text consistent with the rest of the text.
                keep the response short and sweet.'''.format(prompt)
            },
        ],
        stream=True
    )

    collected_chunks = []
    collected_messages = []

    for chunk in completion:
        chunk_time = time.time() - start_time
        collected_chunks.append(chunk)
        chunk_message = chunk.choices[0].delta.content
        collected_messages.append(chunk_message)

    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join([m for m in collected_messages])

    return jsonify({
        'time': chunk_time,
        'conversation': full_reply_content
    })

if __name__ == '__main__':
    app.run(debug=True)