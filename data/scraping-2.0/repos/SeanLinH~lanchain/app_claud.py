from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import os 
import streamlit as st
import asyncio

load_dotenv()

anthropic = AsyncAnthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


async def generate_text(x):
    container = st.empty()
    text = ""
    stream = await anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=512,
        prompt=f"{HUMAN_PROMPT} {x} {AI_PROMPT}",
        stream=True,
        temperature=0.9,
    )
    # print(completion.completion)
    async for completion in stream:
        new_char = completion.completion
        text += new_char
        container.text(text)
        asyncio.sleep(0.1)  
        # print(completion.completion, end="", flush=True)

prompt = st.text_input('士桓AI哥在此')


if prompt:
    asyncio.run(generate_text(prompt))

