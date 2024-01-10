# Standard library imports
import os
import time
import shutil
import tempfile
import wavio
import keyboard
import os
import tempfile
import numpy as np
import openai
import sounddevice as sd
import soundfile as sf
# import tweepy
from elevenlabs import generate, play, set_api_key
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper

# Third party imports
import requests
import numpy as np
import gradio as gr
import assemblyai as aai
import sounddevice as sd
import openai
from scipy.io.wavfile import write
from gradio.components import Audio, Textbox, Radio, Checkbox
from elevenlabs import clone, generate, play, stream, set_api_key
from dotenv import load_dotenv
# from baseapi import *

# Define constants
AUDIO_TYPE_MAPPING = {
    "Conference": "conference",
    "Interview": "interview",
    "Lecture": "lecture",
    "Meeting": "meeting",
    "Mobile Phone": "mobile_phone",
    "Music": "music",
    "Podcast": "podcast",
    "Studio": "studio",
    "Voice Over": "voice_over",
}


cloned_voice = None


# class ChatWrapper:
#     def __init__(self, generate_speech, generate_text):
#         self.lock = Lock()
#         self.generate_speech = generate_speech
#         self.generate_text = generate_text
#         self.s2t_processor_ref = bentoml.models.get("whisper_processor:latest")
#         self.processor = bentoml.transformers.load_model(self.s2t_processor_ref)
#
#     def __call__(
#         self,
#         api_key: str,
#         audio_path: str,
#         text_message: str,
#         history: Optional[Tuple[str, str]],
#         chain: Optional[ConversationChain],
#     ):
#         """Execute the chat functionality."""
#         self.lock.acquire()
#
#         print(f"audio_path : {audio_path} ({type(audio_path)})")
#         print(f"text_message : {text_message} ({type(text_message)})")
#
#         try:
#             if audio_path is None and text_message is not None:
#                 transcription = text_message
#             elif audio_path is not None and text_message in [None, ""]:
#                 audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column(
#                     "audio",
#                     Audio(sampling_rate=16000),
#                 )
#                 sample = audio_dataset[0]["audio"]
#
#                 if sample is not None:
#                     input_features = self.processor(
#                         sample["array"],
#                         sampling_rate=sample["sampling_rate"],
#                         return_tensors="pt",
#                     ).input_features
#
#                     transcription = self.generate_text(input_features)
#                 else:
#                     transcription = None
#                     speech = None
#
#             if transcription is not None:
#                 history = history or []
#                 # If chain is None, that is because no API key was provided.
#                 if chain is None:
#                     response = "Please paste your Open AI key."
#                     history.append((transcription, response))
#                     speech = (PLAYBACK_SAMPLE_RATE, self.generate_speech(response))
#                     return history, history, speech, None, None
#                 # Set OpenAI key
#                 import openai
#
#                 openai.api_key = api_key
#                 # Run chain and append input.
#                 output = chain.run(input=transcription)
#                 speech = (PLAYBACK_SAMPLE_RATE, self.generate_speech(output))
#                 history.append((transcription, output))
#
#         except Exception as e:
#             raise e
#         finally:
#             self.lock.release()
#         return history, history, speech, None, None
#
#
# chat = ChatWrapper(generate_speech, generate_text)


def clone_and_stream_voice(name, description, labels):
    voice = clone(
        name=name, description=description, files=["output.wav"], labels=labels
    )

    return voice


def get_access_token():
    payload = {"grant_type": "client_credentials", "expires_in": 1800}
    response = requests.post(
        "https://api.dolby.io/v1/auth/token",
        data=payload,
        auth=requests.auth.HTTPBasicAuth(APP_KEY, APP_SECRET),
    )
    return response.json()["access_token"]


def upload_media(file_path, headers):
    upload_url = "https://api.dolby.com/media/input"
    upload_body = {"url": f"dlb://in/{os.path.basename(file_path)}"}
    response = requests.post(upload_url, json=upload_body, headers=headers)
    response.raise_for_status()
    presigned_url = response.json()["url"]

    with open(file_path, "rb") as input_file:
        requests.put(presigned_url, data=input_file)


def create_enhancement_job(file_path, output_path, headers, audio_type):
    enhance_url = "https://api.dolby.com/media/enhance"
    enhance_body = {
        "input": f"dlb://in/{os.path.basename(file_path)}",
        "output": f"dlb://out/{os.path.basename(output_path)}",
        "content": {"type": audio_type},
    }
    response = requests.post(enhance_url, json=enhance_body, headers=headers)
    response.raise_for_status()
    return response.json()["job_id"]


def check_job_status(job_id, headers):
    status_url = "https://api.dolby.com/media/enhance"
    params = {"job_id": job_id}
    while True:
        response = requests.get(status_url, params=params, headers=headers)
        response.raise_for_status()
        status = response.json()["status"]
        if status == "Success":
            break
        print(f"Job status: {status}, progress: {response.json()['progress']}%")
        time.sleep(5)


def download_enhanced_file(output_path, headers):
    download_url = "https://api.dolby.com/media/output"
    args = {"url": f"dlb://out/{os.path.basename(output_path)}"}
    with requests.get(
        download_url, params=args, headers=headers, stream=True
    ) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        print(f"Downloading from {response.url} into {output_path}")
        with open(output_path, "wb") as output_file:
            shutil.copyfileobj(response.raw, output_file)


def dolby_process(input_file, output_file, audio_type):
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    upload_media(input_file, headers)
    job_id = create_enhancement_job(input_file, output_file, headers, audio_type)
    check_job_status(job_id, headers)
    download_enhanced_file(output_file, headers)


def enhance_audio(recording, upload, audio_type):
    audio_type = audio_type_mapping[audio_type]
    if recording is not None:
        rate, data = recording
        temp_input_file = "input.wav"
    elif upload is not None:
        rate, data = upload
        if rate not in [44100, 48000] or data.dtype not in [np.int16, np.int32]:
            return None, None, "Invalid file type. Please upload an MP3 file."
        temp_input_file = "input.mp3"
    else:
        return (
            None,
            None,
            "Invalid input. Please record some audio or upload an audio file.",
        )

    write(temp_input_file, rate, data)

    temp_output_file = "output.wav"
    dolby_process(
        temp_input_file, temp_output_file, audio_type
    )  # Pass the audio type to the Dolby processing function

    return temp_input_file, temp_output_file, "Processing complete!"


def clone_voice(temp_output_file):
    # Your voice cloning logic goes here
    cloned_voice_file = "cloned_voice.wav"
    return cloned_voice_file, "Voice cloning complete!"


audio_type_mapping = {
    "Conference": "conference",
    "Interview": "interview",
    "Lecture": "lecture",
    "Meeting": "meeting",
    "Mobile Phone": "mobile_phone",
    "Music": "music",
    "Podcast": "podcast",
    "Studio": "studio",
    "Voice Over": "voice_over",
}

from gradio import Checkbox


def combined_function(
    recording,
    upload,
    audio_type,
    proceed_to_clone,
    name,
    description,
    accent,
    age,
    gender,
    use_case,
    model,
):
    labels = {
        "accent": accent,
        "description": description,
        "age": age,
        "gender": gender,
        "use case": use_case,
    }
    input_file, output_file, _ = enhance_audio(recording, upload, audio_type)
    if proceed_to_clone:
        voice = clone_and_stream_voice(name, description, labels)
    else:
        voice = "Voice cloning not performed."

    cloned_voice = voice
    return voice


def user_text(
    openai_api_key_textbox,
    audio_message,
    text_message,
):
    print(audio_message)
    pass


def user_audio(
    openai_api_key_textbox,
    audio_message,
    text_message,
):
    print(audio_message)
    pass


def generate_audio(llm_text):
    print("My cloned voice ", cloned_voice)
    # audio = generate(
    #     text=llm_text,
    #     voice=cloned_voice,
    #     model="eleven_monolingual_v1",
    # # )
    audio = generate(
        text=llm_text,
        voice="Bella",
        model="eleven_monolingual_v1",
    )

    play(audio)


def transcribe_audio_openai(file_path):
    # Transcribe audio using OpenAI's Whisper ASR system
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def llm_process(transcribed_text):
    llm = OpenAI(
        openai_api_key="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",
        temperature=0,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    zapier = ZapierNLAWrapper(zapier_nla_api_key="sk-ak-jed9SzIRg7R2F5sio6orTRGX5D")
    toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    tools = toolkit.get_tools() + load_tools(["human"])

    agent = initialize_agent(
        tools,
        llm,
        memory=memory,
        agent="conversational-react-description",
        verbose=True,
    )

    assistant_message = agent.run(transcribed_text)
    return assistant_message
    # play_generated_audio(assistant_message)


def new_combined_function(audio, text):
    transcribed_text = None

    if audio is not None:
        # Save the audio to a file
        audio_path = "recorded_audio.wav"
        rate, data = audio
        wavio.write(audio_path, data, rate, sampwidth=2)
        print("Audio saved to:", audio_path)

        transcribed_text = transcribe_audio_openai(audio_path)
        print("Transcribed text:", transcribed_text)

    if text is not None:
        transcribed_text = text

    assistant_msg = llm_process(transcribed_text)
    generate_audio(assistant_msg)

    return assistant_msg


def main():
    iface = gr.Interface(
        fn=combined_function,
        inputs=[
            Audio(source="microphone", label="Recorded Audio"),
            Audio(source="upload", label="Uploaded Audio"),
            Radio(choices=list(audio_type_mapping.keys()), label="Audio Type"),
            Checkbox(label="Proceed to Clone Voice"),
            Textbox(label="Name"),
            Textbox(label="Description"),
            Textbox(label="Accent"),
            Textbox(label="Age"),
            Textbox(label="Gender"),
            Textbox(label="Use Case"),
            Radio(
                choices=["eleven_monolingual_v1", "eleven_multilingual_v1"],
                label="Model",
            ),
        ],
        outputs=Textbox(label="Cloned Voice"),
        title="Audio Enhancer, Transcriber and Voice Cloner",
        description="Enhance your audio, transcribe it and clone voices using the Dolby API",
        allow_flagging="never",
    )

    # iface.launch(inbrowser=True, share=True)

    # iface_2 = gr.Interface(
    #     fn=combined_function,
    #     inputs=[gr.Microphone(label="Speak Your Query")],
    #     outputs=Textbox(label="Cloned Voice"),
    #     title="Agent Vinod",
    #     description="Enhance your audio, transcribe it and clone voices using the Dolby API",
    #     allow_flagging="never",
    # )
    # block = gr.Blocks(css=".gradio-container")

    iface_2 = gr.Interface(
        fn=new_combined_function,
        # openai_api_key_textbox=gr.Textbox(
        #     placeholder="Paste your OpenAI API key (sk-...)",
        #     show_label=False,
        #     lines=1,
        #     type="password",
        # ),
        # chatbot=gr.Chatbot(),
        # audio = gr.Audio(label="Chatbot Voice", elem_id="chatbox_voice"),
        inputs=[
            Audio(
                label="User voice message",
                source="microphone",
            ),
            Textbox(
                label="What do you want to get done ",
                placeholder="Create a spreadsheet for me",
            ),
        ],
        outputs=Textbox(label="Answer by gpt"),
    )

    # openai_api_key_textbox.change(
    #     set_openai_api_key,
    #     inputs=[openai_api_key_textbox],
    #     outputs=[agent_state],
    #     show_progress=False,
    # )

    # iface_2.launch(inbrowser=True, share=True)
    #
    demo = gr.TabbedInterface([iface, iface_2], ["Text-to-speech", "Agent Vinod"])
    demo.launch(share=True)


if __name__ == "__main__":
    main()
