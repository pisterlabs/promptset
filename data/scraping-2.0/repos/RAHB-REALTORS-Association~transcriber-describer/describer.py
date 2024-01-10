import openai
import requests

def generate_description(transcription, model):
    prompt = f"I have a transcript of an audio file and I need a brief description or summary of it. Here's the transcript followed by ENDOFTRANSCRIPT: \"{transcription}\" ENDOFTRANSCRIPT. Can you provide a summary?"

    completion = openai.ChatCompletion.create(
        model=model, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes and describes audio transcripts with no other commentary so that they can be copied into a video description."},
            {"role": "user", "content": prompt}
        ]
    )

    description = completion['choices'][0]['message']['content']
    return description

def generate_description_local(transcription, model):
    prompt = f"I have a transcript of an audio file and I need a brief description or summary of it. Here's the transcript followed by ENDOFTRANSCRIPT: \"{transcription}\" ENDOFTRANSCRIPT. Can you provide a summary?"

    # Send a POST request to the local server
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes and describes audio transcripts with no other commentary so that they can be copied into a video description."},
                {"role": "user", "content": prompt}
            ]
        }
    )

    # Retrieve the description from the response
    description = response.json()['choices'][0]['message']['content']
    return description
