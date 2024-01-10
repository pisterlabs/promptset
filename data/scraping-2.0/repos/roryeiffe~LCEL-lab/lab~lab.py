from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnablePassthrough

import os

# Set Up API access via environment variables:
api_key = os.environ['OPENAI_API_KEY']
base_url = os.environ['OPENAI_API_BASE']
deployment = os.environ['DEPLOYMENT_NAME']
version = os.environ['OPENAI_API_VERSION']

# TODO: Complete this prompt to ask the model for general information on a {topic}:
prompt_template = "{topic}"
prompt = ChatPromptTemplate.from_template(prompt_template)

# Create a model:
model = AzureChatOpenAI(openai_api_version="2023-05-15",azure_deployment=deployment)

# Use a simple output parser that converts output to a string
output_parser = StrOutputParser()

# TODO: Create/return a chain using the prompt, model, and output_parser
# Make sure you use LCEL to achieve this. 
# Hint: The function body can be as short as a single line
def get_basic_chain():
    chain = None
    return chain

# Using the chain created in basic_chain, invoke the chain with a topic.
# PLEASE DO NOT edit this function
def basic_chain_invoke(topic):
    chain = get_basic_chain()
    try:
        response = chain.invoke({"topic": topic})
    except Exception as e:
        return "Something went wrong: {}".format(e)
    return response

# TODO: Complete this prompt so that it asks the model for
# a list of actors that appear in {movie}
movie_prompt = """
    {movie}
"""

# Because we are prompting for a list of actors, use the
# following output parser:
actors_output_parser = CommaSeparatedListOutputParser()

# TODO: Implement the following function. The function should
# return a chain that takes in a movie and returns a list of
# actors who appeared in that movie. 
# Again, make sure to use LCEL to construct the chain
# Ensure that the output key is "actors"
def get_movie_to_actors_chain():
    chain = None
    return chain


# TODO Fill out the prompt so that it asks the model for movies which share at
# least 3 {actors} as the original movie, excluding the original movie.
actor_prompt = """
    "{actors}"
"""

# TODO: Implement the following function. The function should return a chain
# that takes in the actors list from the previous chain and returns a string
# containing movies that share at least 3 common actors (not including the 
# original movie)
# Again, make sure to use LCEL to construct the chain
# To help get you started, some initial code is provided:
def get_actors_to_movies_chain():
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("human","Which actors are in the following movie."),
            ]
        )
    )
    return chain

# TODO: Finally, this function should return a final chain that links
# up the 2 previous chains. When invoking this chain, you should be able
# pass in a movie and have the chain return a list of movies that share similar
# actors
# Again, make sure to use LCEL to construct the chain
def get_final_chain():
    chain = None

    return chain

# This function takes the final chain, invokes it with
# a passed-in movie and returns the response
# PLEASE DO NOT edit this function
def final_chain_invoke(movie):
    chain = get_final_chain()
    try:
        response = chain.invoke({"movie": movie})
        return response
    except Exception as e:
        return "Something went wrong: {}".format(e)
    
