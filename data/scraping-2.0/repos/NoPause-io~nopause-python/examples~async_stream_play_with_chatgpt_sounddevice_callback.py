# Copyright 2023 NoPause

import queue
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
    try:
        # Note: openai key is needed for chatgpt
        text_stream_type = 'chatgpt' # chatgpt | text

        if text_stream_type == 'chatgpt':
            text_agenerator = chatgpt_stream("Hello, who are you?")
        else:
            text_agenerator = text_stream()

        audio_chunks = await nopause.Synthesis.astream(text_agenerator, voice_id="Zoe")

        q = queue.Queue()
        loop = asyncio.get_event_loop()
        event = asyncio.Event() # For non-async, just use threading.Event
        input_done = False

        def callback(outdata, frames, time, status):
            nonlocal input_done
            if q.empty():
                outdata[:] = b'\x00' * len(outdata)
                if input_done:
                    loop.call_soon_threadsafe(event.set)
                return
            chunk_data = q.get()
            # single channel
            outdata[:len(chunk_data)] = chunk_data
            if len(chunk_data) < len(outdata):
                outdata[len(chunk_data):] = b'\x00' * (len(outdata) - len(chunk_data))
            if input_done and q.empty():
                loop.call_soon_threadsafe(event.set)

        samplerate = 24000
        blocksize = 4800
        stream = sd.RawOutputStream(
            samplerate=samplerate, blocksize=blocksize,
            device=sd.query_devices(kind="output")['index'],
            channels=1, dtype='int16',
            callback=callback)

        with stream:
            async for chunk in audio_chunks:
                # Note, a block of int16 (blocksize*1 16-bit) = two blocks of bytes (blocksize*2 8-bit)
                for i in range(0, len(chunk.data), blocksize*2):
                    q.put_nowait(chunk.data[i:i+blocksize*2])
            input_done = True

            await event.wait()
            await asyncio.sleep(1)

        print('Play done.')

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except BaseException as e:
        raise e

if __name__ == '__main__':
    asyncio.run(main())
