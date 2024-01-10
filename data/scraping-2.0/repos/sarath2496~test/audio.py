import os
import sounddevice as sd
import openai
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import tkinter as tk
from tkinter import scrolledtext
import openai
import os
import multion
from langchain.agents.agent_toolkits import MultionToolkit
from langchain.document_loaders import ArxivLoader
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
openai.api_key = os.environ.get('OPENAI_API_KEY')
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
llm = OpenAI(temperature=0)
from langchain.agents.agent_toolkits import MultionToolkit
multion.login()
toolkit = MultionToolkit()
tools=toolkit.get_tools()
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

latest_transcription = ""

def record_audio(duration, samplerate=16000):
    output_text.insert(tk.END, "Recording for {} seconds...\n".format(duration))
    output_text.update()
    recorded_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float64')
    sd.wait()
    output_text.insert(tk.END, "Recording complete!\n")
    output_text.update()
    return recorded_data

def transcribe_audio(audio_data, samplerate):
    processor = WhisperProcessor.from_pretrained("openai/whisper-large")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
    model.config.forced_decoder_ids = None

    input_features = processor(audio_data, sampling_rate=samplerate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, max_new_tokens=500)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    global latest_transcription
    latest_transcription = transcription[0]
    output_text.insert(tk.END, "Transcription: {}\n".format(latest_transcription))
    output_text.update()

def start_recording_and_transcribe():
    recorded_data = record_audio(5)
    transcribe_audio(recorded_data.flatten(), 16000)

def send_to_multion():
    global latest_transcription
    if latest_transcription:
        agent.run(
            f"execute this text '{latest_transcription}'"
            )
    else:
        output_text.insert(tk.END, "No transcription available.\n")
        output_text.update()

# GUI code
root = tk.Tk()
root.title("Audio Transcriber")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

record_btn = tk.Button(frame, text="Start Recording & Transcribe", command=start_recording_and_transcribe)
record_btn.pack(pady=20)

multion_btn = tk.Button(frame, text="Send to Multion", command=send_to_multion)
multion_btn.pack(pady=20)

output_text = scrolledtext.ScrolledText(frame, width=70, height=10)
output_text.pack(pady=20)

root.mainloop()
