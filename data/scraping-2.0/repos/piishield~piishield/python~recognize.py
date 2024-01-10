import openai
import json
import os
import glob
import asyncio
import pystache

async def generate_response(input_text, prompt):
    prompt = pystache.render(prompt, {"sentance": input_text})
    messages = [
        {"role": "system", "content": "You are a privacy preserving AI. Return output only in JSON."},
        {"role": "user", "content": f"{prompt} Input: {input_text} PII:"}
    ]
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=2000,
        temperature=0.1,
        messages=messages
    )
    output = completion['choices'][0]['message']['content']
    return output


def load_prompts(location="../prompts"):
    files = glob.glob(f"{location}/*.hbs")
    prompts = []
    for file in files:
        with open(file, 'r') as f:
            prompts.append(f.read())
    return prompts

def analyze_text(input_text, location="../prompts"):
    prompts = load_prompts(location)

    async def prompt_submit():
        tasks = set()
        for prompt in prompts:
            pii_call = asyncio.create_task(generate_response(input_text, prompt))
            tasks.add(pii_call)
        result = await asyncio.gather(*tasks)
        print(result)
        return result
    results = asyncio.run(prompt_submit())
    return results

def redact_text(input_text):
    results = analyze_text(input_text)
    output_pii = input_text
    for result in results:
        pii_json = json.loads(result)
        print(pii_json)
        category = next(iter(pii_json))
        values = pii_json[category]
        for pii_val in values:
            print(pii_val)
            if pii_val in output_pii:
                output_pii = output_pii.replace(pii_val, f'[{category.upper()[:-1]}]')
    return output_pii

if __name__ == "__main__":
    print(redact_text("Hello! I am amy a girl"))

