from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.llms import HuggingFaceEndpoint


import os

model = HuggingFaceEndpoint(
    endpoint_url=os.environ['LLM_ENDPOINT'],
    huggingfacehub_api_token=os.environ['HF_TOKEN'],
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512
    }
)


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
def get_movie_to_actors_chain():
    chain = None
    return chain


# TODO Fill out the prompt so that it asks the model for movies which share at
# least 1 {actors} as the original movie, excluding the original movie.
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