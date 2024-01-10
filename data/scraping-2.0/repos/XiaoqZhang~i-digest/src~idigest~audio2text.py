import os
import openai
import whisper
from loguru import logger

def a2t(audio_path):
    logger.debug(f"{audio_path}")
    openai.api_key = os.environ['OPENAI_API_KEY']

    model = whisper.load_model('base')
    result = model.transcribe(audio_path, fp16=False)['text']

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    message = "Give a summary of '%s'" %result

    while message:
        messages.append(
            {
                "role": "user",
                "content": message
            }
        )
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        summary = chat_completion.choices[0].message.content

        message = "Come up with questions for '%s'" %result
        messages.append(
            {
                "role": "user",
                "content": message
            }
        )

        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        question = chat_completion.choices[0].message.content

        message="Suggest 3 keywords for '%s'" %result
        messages.append(
            {
                "role": "user",
                "content": message
            }
        ) 
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reference = chat_completion.choices[0].message.content        
        
        message = False

    return result, summary, question, reference