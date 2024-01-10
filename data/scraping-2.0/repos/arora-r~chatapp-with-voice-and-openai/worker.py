import openai
import requests

openai.api_key = "..."


def speech_to_text(audio_binary):
    base_url = 'http://host.docker.internal:1080'
    api_url = base_url + '/speech-to-text/api/v1/recognize'
    params = {'model': 'en-US_Multimedia'}
    body = audio_binary
    response = requests.post(api_url, params=params, data=body).json()
    text = 'null'
    while bool(response.get('results')):
        print('speech to text response:', response)
        text = response.get('results').pop().get(
            'alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text


def text_to_speech(text, voice=""):
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    json_data = {'text': text}

    base_url = 'http://host.docker.internal:1081'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    response = requests.post(api_url, headers=headers, json=json_data)
    print('text to speech response:', response)
    return response.content


def openai_process_message(user_message):
    prompt = "\"Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations. " + user_message + "\""
    print("prompt:", prompt)
    openai_response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=4000)
    print("openai response:", openai_response)
    response_text = openai_response.choices[0].text
    return clean_text(response_text)


def clean_text(text):
    text = text.replace(':', '').replace('\n', '')
    text = ' '.join(text.split())
    return text
