from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Together
from dotenv import load_dotenv
import os
from third_parties import linkedin
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    llm = Together(
        model="DiscoResearch/DiscoLM-mixtral-8x7b-v2",
        temperature=0,
        max_tokens=512,
        top_k=50,
        top_p=1,
        together_api_key=os.getenv("TOGETHER_API_KEY"),
    )
    linkedin_profile_url = linkedin_lookup_agent(name="Eden Marco")

    summary_template = """
        given the Linkedin information: {information} about a person, I want you to create:
        1. A short summary
        2. Two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = linkedin.scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url
    )
    print(chain.run(information=linkedin_data))
