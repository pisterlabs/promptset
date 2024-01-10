import os  # Importing the 'os' module to handle operating system related functionalities
from langchain.prompts import (
    PromptTemplate,
)  # Importing 'PromptTemplate' from a custom module
from langchain.chat_models import (
    ChatOpenAI,
)  # Importing 'ChatOpenAI' from a custom module
from langchain.chains import LLMChain  # Importing 'LLMChain' from a custom module
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from agents.linkedin_lookup_agent import linkedin_lookup_agent
from agents.twitter_lookup_agent import twitter_lookup_agent
from output_parsers import person_intel_parser, PersonIntel
from typing import Tuple

# Fetches the OpenAI API key from the environment variables, providing a default message if not set
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY", "The environment variable 'OPENAI_API_KEY' is not set"
)

name = "Elon Musk"
company = "Tesla"


# Return PersonIntel isntance and profile picture url
def ice_break(name: str, company: str) -> Tuple[PersonIntel, str]:
    # The prompt template containing a placeholder {information}
    summary_template = """
    Given the LinkedIn information '{linkedin_information}' and twitter {twitter_information} about a person, I want you to create:
    1. a short summary
    2. two interesting facts about them
    3. A topic that may interest them
    4. 2 Creative Ice breakers to open a conversation with them
    \n{format_instructions}
    """

    # PromptTemplate contains variables and a template to create prompts
    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information", "twitter_information"],
        template=summary_template,
        # A way for us to provide a subset of the required input variables in the prompt template
        # Injected directly into the prompt template without waiting for the chain to run.
        partial_variables={
            # tell llm that the output should be in that specific format (schema of the pydanticObject is plugged in).
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    # Setting up a language model to be used for generating responses
    # temperature parameter decides how creative the language model will be (0 implies less creativity)

    # gpt-3.5 token limit of 4k
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Chain setup: Using the ChatModel with the specified language model and prompt template
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    linkedin_profile_url = linkedin_lookup_agent(name=name, company=company)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username, num_tweets=100)

    # Running the chain using the user's provided information as a parameter
    result = chain.run(linkedin_information=linkedin_data, twitter_information=tweets)

    # Return us a dictionary / instance of the PersonIntel class
    return (person_intel_parser.parse(result), linkedin_data.get("profile_pic_url"))

    # Now we can run this chain with different parameter values inside the prompt template.


# Only execute if this file was executed directly.
if __name__ == "__main__":
    ice_break(name=name, company=company)
