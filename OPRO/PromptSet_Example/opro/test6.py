from llm_async import run_llm_coroutine
import asyncio
from tqdm import tqdm
import json
import os
import re

CHOSEN_PROMPT = "Make a code review of the changes made in this diff: PLACEHOLDER"

# Generate a question and answer pair using a language model
async def generate_synthetic_data(CHOSEN_PROMPT, sample_size=1):
    async def generate_synthetic_datapoint(request_count):
        SYNTH_DATA_GEN_PROMPT = """You are a helpful assistant designed to generate synthetic data for the prompt: "{CHOSEN_PROMPT}".
Please generate a text and response pair for the prompt. Text is to be interpolated in the prompt. Response is the expected response to the prompt with the text interpolated.
Ensure that the text is delimited by <BEGIN_TEXT> and <END_TEXT> and the response is delimited by <BEGIN_RESPONSE> and <END_RESPONSE>. Generate text and response that is in the format shown below and highly relevant to the prompt. Take a deep breath and think step-by-step.

## Example Format:
<BEGIN_PROMPT> This is the prompt provided <END_PROMPT>
<BEGIN_TEXT> This is the text to be interpotated into the prompt. <END_TEXT>
<BEGIN_RESPONSE> The response to be generated from the text-interpolated prompt. <END_RESPONSE>

## Query:
<BEGIN_PROMPT> {CHOSEN_PROMPT} <END_PROMPT>
"""
        data_pairs = []
        unique_data = set()

        pbar = tqdm(total=request_count, desc="Generating Synthetic Data")
        attempt_count = 0
        while len(data_pairs) < request_count and attempt_count < 50:
            attempt_count += 1
            print(f"Attempt {attempt_count} made.")
            data_gen_prompt = SYNTH_DATA_GEN_PROMPT.format(CHOSEN_PROMPT=CHOSEN_PROMPT)
            response = await run_llm_coroutine([data_gen_prompt for _ in range(request_count)], temperature=1.2, model="llama3-70b", msg="Generating Synthetic Data - 100 calls")
            for res in response:
                print(res)
                try:
                    # Checking if the response is valid
                    text_match = re.search(r"<BEGIN_TEXT>([\s\S]*?)<END_TEXT>", res)
                    response_match = re.search(r"<BEGIN_RESPONSE>([\s\S]*?)<END_RESPONSE>", res)
                    assert text_match is not None and response_match is not None, "Invalid response format."
                    text = text_match.group(1).strip()
                    response = response_match.group(1).strip()
                    assert text not in unique_data, "Data already exists in the set."
                    unique_data.add(text)
                    data_pairs.append({"text": text, "response": response})
                    pbar.update(1)
                except Exception as e:
                    print(e)
                    # print(res)
                    continue
        pbar.close()
        return data_pairs[:request_count]

    # Generating synthetic data
    text = await generate_synthetic_datapoint(sample_size)
    with open("synthetic_data_TEMP.json", "w") as f:
        json.dump(text, f, indent=4)

    return text

if __name__ == "__main__":
    print(asyncio.run(generate_synthetic_data(CHOSEN_PROMPT)))