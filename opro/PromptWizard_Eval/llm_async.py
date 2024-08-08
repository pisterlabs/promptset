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

async def llm_coroutine(prompt, temperature, max_tokens, model, respond_json):
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
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json"} if respond_json else None,
            )
            break
        except APIConnectionError as e:
            print(f"API Connection Error: {e}. Retrying...")
    
    return chat_completion.choices[0].message.content


async def run_llm_coroutine(prompts, temperature=0.0, max_tokens=8192, model="llama3-8b", respond_json=False, msg=None):
    """
    Run the LLM model with the given prompts and temperature. 
    Input: List of prompts, temperature. Output: List of responses.
    """    
    batch = asyncio.gather(*(llm_coroutine(prompt, temperature, max_tokens, model, respond_json) for prompt in prompts))
    responses = await batch
        
    return responses
