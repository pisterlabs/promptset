from flask import Flask, render_template, request, jsonify
import os
import openai
from dotenv import load_dotenv
import os
import numpy as np
import json
import re
import os
import gunicorn
import requests
import logging
import pandas as pd

#test
# test 2
# test 3

load_dotenv()
# OPENAI_API_KEY = os.environ.get('OPEN_AI_KEY')
app = Flask(__name__)

if not os.path.exists('recorded_audio'):
    os.makedirs('recorded_audio')


# def get_openai_key():
#     # Step 1: Check if OPEN_AI_KEY exists in environment variables
#     OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
#     return True


@app.route('/', methods=['GET', 'POST'])
def index():
    # Retrieve OPENAI_API_KEY
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    # Step 2: If not found, test if token is provided by the textfield
    if OPENAI_API_KEY is None and request.method == 'POST':
        OPENAI_API_KEY = request.form.get('openai-token')

    # If found, set the API key
    if OPENAI_API_KEY:
        # Step 3: After user provided token, send success msg and set token as OPENAI_API_KEY
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY

        # Render the template with a success message
        return render_template('audio.html', user_message="Token set successfully!")
    else:
        # Render the template with an error message to prompt user to provide the token
        return render_template('audio.html', user_message="You need to provide the OpenAI token in order to continue.")



@app.route('/impressum', methods=['GET', 'POST'])
def impressum():
    return render_template('impressum.html')


def get_embedding(text, model="text-embedding-ada-002"):
    openai.api_key = os.environ.get('OPEN_AI_KEY')
    text = text.replace("\n", " ")
    logging.info(text)
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def generate_text(prompt, freshness, frequency_penalty, max_tokens, model_id):
    OPENAI_API_URL = "https://api.openai.com/v1/engines/" + model_id + "/completions"
    data = {
        'prompt': prompt,
        'temperature': float(freshness),
        'frequency_penalty': float(frequency_penalty),
        'max_tokens': int(max_tokens),
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai.api_key}',
    }
    response = requests.post(OPENAI_API_URL, json=data, headers=headers)
    if response.status_code != 200:
        return {'error': 'Failed to generate text'}
    try:
        response_data = response.json()
        choices = response_data['choices']
        text = choices[0]['text']
        unwanted_characters = r'[@#€]'  # Add any other unwanted characters inside the brackets
        text = re.sub(unwanted_characters, '', text)
        text = re.sub(r'\n+', '\n', text)  # Remove consecutive occurrences of '\n'

        # Get embeddings for the prompt, text (completion), and concatenated text
        # prompt_embedding = get_embedding(prompt)
        # text_embedding = get_embedding(text)
        # concat_text = prompt + " " + text
        # concat_text_embedding = get_embedding(concat_text)
        #
        # # Save the information in a pandas DataFrame
        # df = pd.DataFrame(columns=['prompt_embedding', 'text_embedding', 'concat_text_embedding', 'concat_text'])
        # df = df.append({
        #     'prompt_embedding': prompt_embedding,
        #     'text_embedding': text_embedding,
        #     'concat_text_embedding': concat_text_embedding,
        #     'concat_text': concat_text
        # }, ignore_index=True)
        #
        # df.to_csv('embeddings.csv')

        # graph_data = create_graph_data(prompt, text)
        # graph_data_json = json.dumps(graph_data)

        return text
    except KeyError:
        return {'error': 'Invalid response from OpenAI'}


def create_graph_data(prompt, sentences, translations):
    nodes = [{'id': 1, 'label': prompt}]
    edges = []

    for idx, (sentence, translation) in enumerate(zip(sentences, translations), start=1):
        sentence_node_idx = 2 * idx
        translation_node_idx = 2 * idx + 1

        nodes.append({'id': sentence_node_idx, 'label': sentence})
        nodes.append({'id': translation_node_idx, 'label': translation})

        if sentence_node_idx == 2:
            edges.append({'from': 1, 'to': sentence_node_idx})
        else:
            edges.append({'from': sentence_node_idx - 2, 'to': sentence_node_idx})

        edges.append({'from': sentence_node_idx, 'to': translation_node_idx})

    return {'nodes': nodes, 'edges': edges}


@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    audio_file = request.files['audio']
    translation_prompt_template = request.form['translation-prompt']  # Get the custom translation prompt
    audio_path = 'recorded_audio/last_recorded_audio.wav'
    audio_file.save(audio_path)

    # Transcribe using OpenAI
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    text = transcript["text"]
    print(f"Transcribed text: {text}")

    # Split the text into sentences
    sentences = re.split(r'[.,!?;:-]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    print(f"Extracted sentences: {sentences}")

    # Translate the sentences
    translations = []
    for sentence in sentences:
        translation_prompt = translation_prompt_template.format(sentence=sentence)
        print(f"Translation prompt: {translation_prompt}")

        freshness = float(request.form['freshness'])
        frequency_penalty = float(request.form['frequency-penalty'])
        max_tokens = float(request.form['max_tokens'])


        translation = generate_text(f'{translation_prompt}: {sentence}', freshness, frequency_penalty,
                                    max_tokens,
                                    model_id='text-davinci-003')

        # translation = generate_text(f'{translation_prompt}: {sentences}', freshness=0.8, frequency_penalty=0.0,
        #                             max_tokens=60,
        #                             model_id='text-davinci-003')
        translations.append(translation)
        print(f"Translation: {translation}")

    # Create graph data based on transcribed text and translations
    graph_data = create_graph_data("Transcription", sentences, translations)

    transcription_json = {"text": text, "graph_data": graph_data}

    # Write to JSON file
    json_path = 'recorded_audio/transcription.json'
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(transcription_json, json_file, ensure_ascii=False, indent=4)

    print(transcription_json)
    return jsonify(transcription_json)


if __name__ == "__main__":
    app.run(debug=True)
