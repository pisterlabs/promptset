from openai import AsyncOpenAI, APIConnectionError
import asyncio
import time
import os, dotenv
dotenv.load_dotenv()

MODEL_TO_MODELID = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
}
client = AsyncOpenAI(
    api_key=os.getenv("DEEP_INFRA_API"),
    base_url="https://api.deepinfra.com/v1/openai",
)

async def llm_coroutine(prompt, temperature, model):
    while True:
        try:
            chat_completion = await client.chat.completions.create(
                model=MODEL_TO_MODELID[model],
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=8192,
                temperature=temperature,
            )
            break
        except APIConnectionError as e:
            print(f"API Connection Error: {e}. Retrying...")
    return chat_completion.choices[0].message.content


async def run_llm_coroutine(prompts, temperature=0.0, model="llama3-8b"):
    """
    Run the LLM model with the given prompts and temperature. 
    Input: List of prompts, temperature. Output: List of responses.
    """
    batch = asyncio.gather(*(llm_coroutine(prompt, temperature, model) for prompt in prompts))
    responses = await batch
    return responses
