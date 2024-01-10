import openai
import os
import re

def output(prompt):

    openai.api_key = os.environ["OPENAI_TOKEN"]


    # Set the model and prompt
    model_engine = "text-davinci-003"

    max_tokenG = 4096 # max for davinci-003

    try:
        prompt_text = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=5000,
            n=1,
            stop=None,
            temperature=0.5,
        )
    except Exception as e:
        match = re.search(r'\((\d+) in your prompt[\);]?', str(e))
        max_tokenG = max_tokenG - int(match.group(1))


    # Get the number of tokens in the generated text
    generated_text = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokenG,
        n=1,
        stop=None,
        temperature=1,
    ).choices[0].text
    #generated_tokens = len(generated_text.split())
    #print(generated_text.strip())
    return generated_text.strip()
