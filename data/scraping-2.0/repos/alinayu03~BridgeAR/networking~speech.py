import openai

def transcribe(fp: str, whisper_model):
    result = whisper_model.transcribe(fp)
    transcript = result["text"]
    print(transcript)
    return transcript

def translate(transcript: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a translator. Translate these sentences into English."},
            {"role": "user", "content": transcript}
        ]
    )
    translation = response['choices'][0]['message']['content']
    return translation
