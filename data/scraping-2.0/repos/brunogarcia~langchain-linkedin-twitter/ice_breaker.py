from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties import linkedin, twitter
from agents import linkedin_agent, twitter_agent
from tools import output_parser


def ice_break(name: str) -> output_parser.SummaryPerson:
    linkedin_profile_url = linkedin_agent.lookup(name=name)
    linkedin_data = linkedin.scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url
    )

    twitter_username = twitter_agent.lookup(name=name)
    twitter_data = twitter.scrape_user_tweets(username=twitter_username, num_tweets=5)

    summary_template = """
        Given the Linkedin information {linkedin_information} and Twitter {twitter_information} about a person
        I want you to create:
        1. A short summary of the person
        2. Two interesting facts about the them
        3. A topic that may interest them
        4. Two creative ice breakers to open a conversation with them
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["linkedin_information", "twitter_information"],
        partial_variables={
            "format_instructions": output_parser.person_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # type: ignore

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    result = chain.run(
        linkedin_information=linkedin_data,
        twitter_information=twitter_data,
    )

    return output_parser.person_parser.parse(result)


if __name__ == "__main__":
    ice_break(name="Bruno Garcia Echegaray")
