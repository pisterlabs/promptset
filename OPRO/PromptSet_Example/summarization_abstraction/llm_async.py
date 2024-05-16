from openai import AsyncOpenAI, OpenAI
import asyncio
import time
import os, dotenv
dotenv.load_dotenv()

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
client = AsyncOpenAI(
    api_key=os.getenv("DEEP_INFRA_API"),
    base_url="https://api.deepinfra.com/v1/openai",
)

async def llm_coroutine(prompt, temperature):
    chat_completion = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=2048,
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content


async def run_llm_coroutine(prompts, temperature):
    batch = asyncio.gather(*(llm_coroutine(prompt, temperature) for prompt in prompts))
    responses = await batch
    return responses

def run_llm(prompts, temperature=0.0):
    """
    Run the LLM model with the given prompts and temperature. 
    Input: List of prompts, temperature. Output: List of responses.
    """
    return asyncio.run(run_llm_coroutine(prompts, temperature), debug=True)