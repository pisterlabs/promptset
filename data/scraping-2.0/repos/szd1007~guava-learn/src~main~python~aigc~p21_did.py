import openai, os, time, requests
import gradio as gr
from gradio import HTML
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]

memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)
conversation = ConversationChain(
    llm=OpenAI(max_tokens=2048, temperature=0.5),
    memory=memory,
)

avatar_url = "https://cdn.discordapp.com/attachments/1065596492796153856/1095617463112187984/John_Carmack_Potrait_668a7a8d-1bb0-427d-8655-d32517f6583d.png"
avatar_url = "https://scpic.chinaz.net/files/default/imgs/2023-05-17/8a1bc643f3dcb0e9.jpg"

def generate_talk(input, avatar_url,
                  voice_type="microsoft",
                  voice_id="zh-CN-XiaoxiaoNeural",
                  api_key=os.environ.get('DID_API_KEY')):
    url = "https://api.d-id.com/talks"
    payload = {
        "script": {
            "type": "text",
            "provider": {
                "type": voice_type,
                "voice_id": voice_id
            },
            "ssml": "false",
            "input": input
        },
        "config": {
            "fluent": "false",
            "pad_audio": "0.0"
        },
        "source_url": avatar_url
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Basic " + api_key
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def get_a_talk(id, api_key=os.environ.get('DID_API_KEY')):
    url = "https://api.d-id.com/talks/" + id
    headers = {
        "accept": "application/json",
        "authorization": "Basic " + api_key
    }
    response = requests.get(url, headers=headers)
    print(response)
    return response.json()


def get_mp4_video(input, avatar_url=avatar_url):
    print("get_mp4_video")
    response = generate_talk(input=input, avatar_url=avatar_url)
    talk = get_a_talk(response['id'])
    video_url = ""
    index = 0
    while index < 30:
        index += 1
        if 'result_url' in talk:
            video_url = talk['result_url']
            return video_url
        else:
            time.sleep(1)
            talk = get_a_talk(response['id'])
    return video_url


def predict(input, history=[]):
    if input is not None:
        history.append(input)
        response = conversation.predict(input=input)
        video_url = get_mp4_video(input=response, avatar_url=avatar_url)

        video_html = f"""<video width="320" height="240" controls autoplay><source src="{video_url}" type="video/mp4"></video>"""
        history.append(response)
        responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
        return responses, video_html, history
    else:
        video_html = f'<img src="{avatar_url}" width="320" height="240" alt="John Carmack">'
        responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
        return responses, video_html, history


def transcribe(audio):
    os.rename(audio, audio + '.wav')
    audio_file = open(audio + '.wav', "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt="这是一段简体中文的问题。")
    return transcript['text']


def process_audio(audio, history=[]):
    if audio is not None:
        text = transcribe(audio)
        return predict(text, history)
    else:
        text = None
        return predict(text, history)


with gr.Blocks(css="#chatbot{height:500px} .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

    with gr.Row():
        audio = gr.Audio(source="microphone", type="filepath")

    with gr.Row():
        video = gr.HTML(f'<img src="{avatar_url}" width="320" height="240" alt="John Carmack">', live=False)

    txt.submit(predict, [txt, state], [chatbot, video, state])
    audio.change(process_audio, [audio, state], [chatbot, video, state])

demo.launch()