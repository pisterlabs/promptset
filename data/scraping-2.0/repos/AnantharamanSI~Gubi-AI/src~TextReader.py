import openai
import os
import time
import streamlit
import base64

def math2text(client, text):
    context = """
    Convert the following math into spoken words. Only return the spoken words and nothing else.
    """

    messages = [ {"role": "system", "content": ""} ]

    messages.append(
        {"role": "user", "content": context + "\"" + text + "\""})
    
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages
    )

    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print("math reply", reply)
    return reply

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay hidden>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        streamlit.markdown(md, unsafe_allow_html=True)

def talk(text):
    client = openai.OpenAI(api_key=str(os.environ.get("OPENAI_API_KEY")))
    text = math2text(client, text)
    res = client.audio.speech.create(
        model = "tts-1",    
        voice = "nova",
        input = text
    )
    res.stream_to_file("test.mp3")
    print("playing...")
    autoplay_audio("test.mp3")