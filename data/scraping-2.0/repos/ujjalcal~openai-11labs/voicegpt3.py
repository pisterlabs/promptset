import gradio as gr
import openai
from elevenlabs import generate, stream, set_api_key
import time
import asyncio
from functools import partial


from config import OPENAI_API_KEY, ELEVENLABS_API_KEY

# Set API keys
openai.api_key = OPENAI_API_KEY
set_api_key(ELEVENLABS_API_KEY)


messages = ["You are a mortgage ai assistant. Your name is Grace. Please respond to all input in 25 words or less."]

def text_chunks(input_text, chunk_size=25):
    words = input_text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i+chunk_size])

async def gpt3_streamed_response(input_text):
    for chunk in text_chunks(input_text):
        loop = asyncio.get_event_loop()
        
        completion_fn = partial(openai.Completion.create, 
                                engine="text-davinci-003",
                                prompt=chunk,
                                max_tokens=80,
                                n=1,
                                stop=None,
                                temperature=0.5)
        
        response = await loop.run_in_executor(None, completion_fn)
        
        yield response["choices"][0]["text"]

async def audio_stream(text):
    async for audio_data in generate(
        text=text,
        voice="Bella",
        model="eleven_monolingual_v1",
        stream=True
    ):
        return audio_data


async def transcribe(audio):
    global messages

    audio_file = open(audio, "rb")
    
    loop = asyncio.get_event_loop()
    transcript = await loop.run_in_executor(None, openai.Audio.transcribe, "whisper-1", audio_file)
    messages.append(f"\nUser: {transcript['text']}")

    # Get GPT-3 responses in a streaming manner.
    async for response_text in gpt3_streamed_response(transcript['text']):
        audio_data = await audio_stream(response_text)
        stream(audio_data)
        messages.append(response_text)

    chat_transcript = "\n".join(messages)
    return chat_transcript

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath", placeholder="Please start speaking...", duration=10),
    outputs="text",
    title="🤖 My Desktop ChatGPT Assistant 🤖",
    description="🌟 Please ask me your question and I will respond both verbally and in text to you...",
    live=True
)

iface.launch()
