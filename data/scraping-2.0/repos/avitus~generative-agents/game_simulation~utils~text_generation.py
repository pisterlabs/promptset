import openai
import re
import os
from dotenv import load_dotenv
from transformers import pipeline
import baseten

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate(prompt, use_openai=True):
    """
    Generates a text completion for a given prompt using either the OpenAI GPT-3 API or the Hugging Face GPT-3 model.
    
    Args:
    - prompt (str): The text prompt to generate a completion for.
    - use_openai (bool): A boolean flag indicating whether to use the OpenAI API (True) or the Hugging Face GPT-3 model (False).
    
    Returns:
    - str: The generated text completion.
    """
    if use_openai:
        model_engine = "text-davinci-003"
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        message = response.choices[0].text
        return message.strip()

    else:

        # Falcon7 model running on BaseTen
        # {'data': {'generated_text': "### Instruction: You are Nellie Stonehill. ### Response: 7:00: I'm awake and going to breakfast."}}
        model = baseten.deployed_model_id('XP9veEq')
        output = model.predict({
            "prompt": prompt,
            "do_sample": True,
            "max_new_tokens": 300,
        })

        out = output['data']['generated_text']
        if '### Response:' in out:
            out = out.split('### Response:')[1]
        return out.strip()

        # hf_generator = pipeline('text-generation', model='huggyllama/llama-65b', device=0)  <-- didn't work
        # hf_generator = pipeline('text-generation', model='vicgalle/gpt2-alpaca-gpt4', device=0)  <-- buggy
        # hf_generator = pipeline('text-generation', model='lmsys/vicuna-13b-delta-v1.1', device=1)  <-- download killed
        # hf_generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)
        # hf_generator = pipeline(model="declare-lab/flan-alpaca-gpt4-xl", device=0)
        # output = hf_generator(prompt, max_length=len(prompt)+128, do_sample=True)
        
        # out = output[0]['generated_text']
        # if '### Response:' in out:
        #     out = out.split('### Response:')[1]
        # # if '### Instruction:' in out:
        # #     out = out.split('### Instruction:')[0]
        # return out.strip()

def get_rating(x):
    """
    Extracts a rating from a string.
    
    Args:
    - x (str): The string to extract the rating from.
    
    Returns:
    - int: The rating extracted from the string, or None if no rating is found.
    """
    nums = [int(i) for i in re.findall(r'\d+', x)]
    if len(nums)>0:
        return min(nums)
    else:
        return None

# Summarize simulation loop with OpenAI GPT-4
def summarize_simulation(log_output):
    prompt = f"Summarize the simulation loop:\n\n{log_output}"
    response = generate(prompt)
    return response

# Returns an emojii representation of a str
def emojii_repr(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = "colored emojii representation of" + text
    response = openai.Completion.create(
        engine="text-davinci-003",  # Specify the GPT model
        prompt=prompt,
        max_tokens=5,  # Adjust the desired length of the generated response
        n = 1, # Adjust the number of responses to receive
        stop=None,  # Specify a stopping condition, if desired
    )

    # Extract the generated response from the API response
    generated_text = response.choices[0].text.strip()

    # Return the generated response
    return generated_text


