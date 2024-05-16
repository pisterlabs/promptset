from openai import AsyncOpenAI, OpenAI
import asyncio
import time
import os, dotenv

dotenv.load_dotenv()

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

client_sync = OpenAI(
    api_key=os.getenv("DEEP_INFRA_API"),
    base_url="https://api.deepinfra.com/v1/openai",
)

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
        max_tokens=100,
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content


async def run_llm_coroutine(prompts, temperature):
    batch = asyncio.gather(*(llm_coroutine(prompt, temperature) for prompt in prompts))
    responses = await batch
    return responses

def run_llm(prompts, temperature=0.0):
    return asyncio.run(run_llm_coroutine(prompts, temperature))

if __name__ == "__main__":
    prompts = [
        "Say this is a test",
        "What's the meaning of life?",
        "What's the weather like?",
        "What is the capital of France?",
        "Write a detailed summary of the book 'The Great Gatsby' by F. Scott Fitzgerald."
    ]
    start = time.time()
    for prompt in prompts:
        chat_completion = client_sync.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=100,
            temperature=0.0,
        )
        chat_completion.choices[0].message.content
    end = time.time()
    print("Time taken for synchronous requests:", end - start)

    start = time.time()
    run_llm(prompts, 0.0)
    end = time.time()
    print("Time taken for asynchronous requests:", end - start)
