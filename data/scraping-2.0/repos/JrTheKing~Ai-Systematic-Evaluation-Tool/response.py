# The 'os' module in Python provides functions for interacting with the operating system.

import os

# The 'dotenv' module allows you to specify environment variables in a .env file.

from dotenv import load_dotenv

# 'openai' is the Python client library for the OpenAI API. It allows you to interact with the API for tasks like generating text.

import openai

# 'transformers' is a state-of-the-art Natural Language Processing library for training and deploying transformers.

# Here, we import specific classes for the GPT-2 model.

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# load_dotenv() loads environment variables from a .env file into the shell’s environment variables.

load_dotenv()

# Here, the API key for OpenAI is obtained from the environment variables. 

# The key is needed to make requests to the OpenAI API.

OPENAI_KEY = os.getenv("OPENAI_KEY")

# Sets the API key for the 'openai' library.

openai.api_key = OPENAI_KEY

def load_model(model_name):

    """

    Loads a pre-trained AI model and its associated tokenizer.

    

    Parameters:

    model_name: A string specifying the name of the pre-trained model.

    """

    # GPT2LMHeadModel.from_pretrained() loads the pre-trained model specified by 'model_name'.

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT2Tokenizer.from_pretrained() loads the pre-trained tokenizer associated with the model.

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # The function returns the loaded model and tokenizer.

    return model, tokenizer

def generate_response(model, tokenizer, prompt):

    """

    Generates a response from the AI model for a given input prompt and returns attention scores.

    Parameters:

    model: The pre-trained AI model.

    tokenizer: The tokenizer associated with the model.

    prompt: A string that contains the input for the AI model.

    """

    # The input prompt and the end-of-string token are encoded to be fed into the model. 

    # The result is a PyTorch tensor (specified by return_tensors='pt').

    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # Generate a response from the model using the encoded inputs. 

    # The model generates a response up to a maximum length of 500 tokens. It uses sampling (specified by do_sample=True) with a temperature of 0.6. 

    # The output_scores=True argument means that the output will include the attention scores.

    outputs = model.generate(inputs, max_length=500, do_sample=True, temperature=0.6, output_scores=True)

    # The generated response is decoded into a string, and special tokens (like the end-of-string token) are removed.

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # The attention scores are retrieved from the outputs.

    attention_scores = outputs[-1]

    # The function returns the generated response and the attention scores.

    return response, attention_scores
