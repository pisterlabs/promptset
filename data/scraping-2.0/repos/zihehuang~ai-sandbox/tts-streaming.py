import os
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionMessageParam
# import openai chat completion message ParamSpec
from elevenlabs import generate, stream
from typing import Iterator

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

messages:list[ChatCompletionMessageParam]=[
    ChatCompletionSystemMessageParam(role='system',content="You are a pirate.")
]

def write(prompt) -> Iterator[str]:
    messages.append(ChatCompletionUserMessageParam(role='user',content=prompt))
    stream=client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo-1106",
        max_tokens=256,
        stream=True,
    )

    for chunk in stream:
        if (text_chunk := chunk.choices[0].delta.content) is not None:
            yield text_chunk

# text_stream = dummy_write("Hello, how are you?")


userInput = input('What can a pirate do for you? ')
audio_stream = generate(
    text=write(userInput),
    voice='blGoCgKMN3bayNEUxHYo',
    model='eleven_monolingual_v1',
    stream=True,
)

print(type(audio_stream))
stream(audio_stream)
