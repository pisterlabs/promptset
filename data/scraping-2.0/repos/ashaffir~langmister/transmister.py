import os
import gradio as gr
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
messages = [
    {
        "role": "system",
        "content": "you are a spanish speaking tutor.speak to me in Spanish only, and correct my grammer, text if needed. Respond to all input in 50 words or less. Speak in the first person. Do not use quotation marks. Do not say you are an AI language model.",
    }
]


def transcribe(audio):
    global messages, question_df

    audio_filename_with_extension = audio + ".wav"
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="he")

    user_text = f"{transcript['text']}"
    messages.append({"role": "", "content": user_text})

    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    # return chat_transcript
    return chat_transcript


def export_to_file(text):
    with open("tarns.txt", "w") as file:
        file.write(text)


# set a custom theme
theme = gr.themes.Default().set(
    body_background_fill="#000000",
)

with gr.Blocks(theme=theme, title="Hebrew Text Transcriber") as ui:
    audio_input = gr.Audio(source="microphone", type="filepath")

    # text transcript output and audio
    text_output = gr.Textbox(label="Conversation Transcript")

    run = gr.Button("Run")
    run.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[text_output],
    )

    save = gr.Button("Save")
    save.click(fn=export_to_file, inputs=text_output, show_progress=True)

ui.launch(debug=True, share=True)
