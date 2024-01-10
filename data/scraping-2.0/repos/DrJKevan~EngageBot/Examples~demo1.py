import os
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Specify the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# Load the .env file
load_dotenv(dotenv_path)

# Set the OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI Class
llm = OpenAI(openai_api_key=api_key, temperature=0.7)

# First Prompt
template = """
Provide me a summary on {topic}
"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template = template,
)

# Second Prompt
template2 = """
Reduce this text to 50 words:

{topic_description}
"""

second_prompt = PromptTemplate(
    input_variables=["topic_description"],
    template = template2,
)

# Chain prompts together
chain = LLMChain(llm=llm, prompt=prompt)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
explanation = overall_chain.run("Deciding which pokemon to start with")

# Old print statements for reference
# print(prompt.format(topic="Deciding which pokemon to start with"))
# print(chain.run("Deciding which pokemon to start with"))

# Execute the chain and log errors
#try:
#    responses = chain.execute()
#    
    # Print the responses
#    for response in responses:
#        print(response)
#except Exception as e:
#    logging.error("Exception occurred", exc_info=True)
#    print("An error occurred.")

