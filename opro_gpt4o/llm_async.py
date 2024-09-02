from openai import AsyncOpenAI, APIConnectionError
import asyncio
import time
import os, dotenv
dotenv.load_dotenv()
import time

DEBUG = False

MODEL_TO_MODELID = {
    "phi3-14b": "microsoft/Phi-3-medium-4k-instruct",
    "gemma-7b": "google/gemma-1.1-7b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
}
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
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
                response_format={ "type": "json_object" } if respond_json else None,
            )
            break
        except APIConnectionError as e:
            print(f"API Connection Error: {e}. Retrying...")
    
    # DEBUG: Print token usage
    usage = None
    if DEBUG:
        usage = {"input": chat_completion.usage.prompt_tokens, "output": chat_completion.usage.completion_tokens, "cost_estimate": chat_completion.usage.estimated_cost}
        # print(f"Input: {chat_completion.usage.prompt_tokens}; Output: {chat_completion.usage.completion_tokens}, Cost Estimate: {chat_completion.usage.estimated_cost}")
        with open("log.txt", "a") as f:
            f.write(f"$$$ Input: {chat_completion.usage.prompt_tokens}; Output: {chat_completion.usage.completion_tokens}, Cost Estimate: {chat_completion.usage.estimated_cost} $$$\n")

    return chat_completion.choices[0].message.content, usage


async def run_llm_coroutine(prompts, temperature=0.0, max_tokens=4096, model="llama3-8b", respond_json=False, msg=None):
    """
    Run the LLM model with the given prompts and temperature. 
    Input: List of prompts, temperature. Output: List of responses.
    """
    # Run the LLM model with the given prompts
    if DEBUG:
        # print(f"###### Model: {model} | Temperature: {temperature} | Max Tokens: {max_tokens} | Respond JSON: {respond_json} ######")
        with open("log.txt", "a") as f:
            f.write(f"###### Model: {model} | Temperature: {temperature} | Max Tokens: {max_tokens} | Respond JSON: {respond_json} ###### - Msg: {msg} \n")
    
    if type(temperature) == list:
        assert len(temperature) == len(prompts), "Length of temperature list should be equal to the length of prompts list."
    else:
        temperature = [temperature] * len(prompts)
    batch = asyncio.gather(*(llm_coroutine(p, t, max_tokens, model, respond_json) for p,t in zip(prompts, temperature)))
    responses = await batch
    
    output = [res for res, _ in responses]
    
    if DEBUG:
        total_input = 0
        total_output = 0
        total_cost_estimate = 0
        for _, usage in responses:
            total_input += usage["input"]
            total_output += usage["output"]
            total_cost_estimate += usage["cost_estimate"]
        # print(f"Total Input: {total_input}; Total Output: {total_output}; Total Cost Estimate: {total_cost_estimate}")
        with open("log.txt", "a") as f:
            f.write(f"Total Input: {total_input}; Total Output: {total_output}; Total Cost Estimate: {total_cost_estimate}\n")
    
    
    # # Log the responses
    # delimiter = "\n" + "$" * 100 + "\n"
    # with open("log.txt", "a") as f:
    #     f.write(delimiter.join(responses))
    return output
