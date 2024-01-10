# Copyright 2023 NoPause

import asyncio
import openai
import nopause
import sounddevice as sd

# Install sdk packages first:
#      pip install openai nopause

# Install sounddevice (see https://pypi.org/project/sounddevice/)
#      pip install sounddevice

openai.api_key = "your_openai_api_key_here"
nopause.api_key = "your_nopause_api_key_here"

async def chatgpt_stream(prompt: str):
    responses = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo-0613",
        messages=[
                {"role": "system", "content": "You are a helpful assistant from NoPause IO."},
                {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    print("[User]: {}".format(prompt))
    print("[Assistant]: ", end='', flush=True)
    async def agenerator():
        async for response in responses:
            content = response["choices"][0]["delta"].get("content", '')
            print(content, end='', flush=True)
            yield content
        print()
    return agenerator()

async def text_stream():
    sentence = "Hello, how are you?"
    print("[Text]: ", end='', flush=True)
    for char in sentence:
        await asyncio.sleep(0.01) # simulate streaming text and avoid blocking
        print(char, end='', flush=True)
        yield char
    print()

async def main():
    # Note: openai key is needed for chatgpt
    text_stream_type = 'chatgpt' # chatgpt | text

    if text_stream_type == 'chatgpt':
        text_agenerator = chatgpt_stream("Hello, who are you?")
    else:
        text_agenerator = text_stream()

    audio_chunks = await nopause.Synthesis.astream(text_agenerator, voice_id="Zoe")

    stream = sd.RawOutputStream(
        samplerate=24000, blocksize=4800,
        device=sd.query_devices(kind="output")['index'],
        channels=1, dtype='int16',
        )

    with stream:
        async for chunk in audio_chunks:
            stream.write(chunk.data)
        await asyncio.sleep(1)

    print('Play done.')

if __name__ == '__main__':
    asyncio.run(main())
