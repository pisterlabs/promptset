import logging
import random
import os

import openai
from easygpt import EasyGPT, __version__ as easygpt_version

# Initialize logging
logging.basicConfig(level=logging.INFO)
print("EasyGPT version: ", easygpt_version)

# Initialize API key from environment and other variables
openai.api_key = os.environ.get('OPENAI_API_KEY')

system_msg = "You are a school teacher and can speak with a 5 y.o. kid."
temperature = 0.7

# List of models to choose from
models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]

# List of serious questions to ask the model
questions = [
    "What is the meaning of life?",
    "How can we solve world hunger?",
    "Is there a multiverse?",
    "What's the origin of the universe?"
]

# Loop through the list of questions
for question in questions:
    # Randomly select a model for each iteration
    model_name = random.choice(models)
    
    # Initialize the EasyGPT instance
    easygpt_instance = EasyGPT(openai, model_name, system_msg=system_msg, temperature=temperature)

    # Ask the question
    assistant_message, input_price, output_price = easygpt_instance.ask(question)
    
    # Logging the details
    # logging.info(f"Used model: {model_name}")
    # logging.info(f"Assistant's Response: {assistant_message}")
    # logging.info(f"Input Tokens Cost: ${input_price}")
    # logging.info(f"Output Tokens Cost: ${output_price}")
    
    # Displaying conversation in the console
    print(f"You: {question}")
    print(f"Assistant ({model_name}): {assistant_message}")
    print(f"Input Tokens: ${input_price}, Output Tokens: ${output_price}")
    print("=" * 30)
