import openai
import os
import tempfile
from gtts import gTTS
from playsound import playsound


# Token valido por 24h acesse esse site https://platform.openai.com/account/api-keys
openai.api_key = "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"


def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='pt', tld='com.br', slow=False)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name + '.mp3'
        tts.save(temp_file)
        playsound(temp_file)
        os.remove(temp_file)
    except Exception as e:
        print('Something went wrong in text_to_speech:', e)


def ask_question(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        n=1,
        temperature=0.5,
    )

    answer = response.choices[0].text
    return answer


if __name__ == '__main__':
    while True:
        user_input = input("Você: ")
        response = ask_question(user_input)
        print("Chatbot: ", response)
        text_to_speech(response)
