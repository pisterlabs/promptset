from RealtimeTTS import TextToAudioStream, OpenAIEngine, CoquiEngine
import os
from openai import AsyncOpenAI
client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
def dummy_generator():
    yield "Hey guys! "
    yield "These here are "
    yield "realtime spoken words "
    yield "based on openai "
    yield "tts text synthesis."

async def stream_generator():
    stream = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell a joke."}],
            stream=True,
    )
    async for chunk in stream:
        yield chunk.choices[0].delta.content or ""

engine = OpenAIEngine(model="tts-1", voice="onyx")
engine2 = CoquiEngine()
stream = TextToAudioStream(engine2)
stream.feed(stream_generator())
print ("Synthesizing...")
stream.play()