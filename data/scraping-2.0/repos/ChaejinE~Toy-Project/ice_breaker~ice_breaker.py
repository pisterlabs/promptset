from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrap_linkedin_profile, scrap_linkedin_profile_origin
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parser import person_intel_parser, PersonIntel
from typing import Tuple

information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and investor. He is the wealthiest person in the world, with an estimated net worth of US$222 billion as of December 2023, according to the Bloomberg Billionaires Index, and $244 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6] He is the founder, chairman, CEO, and chief technology officer of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation.

A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.

In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, he founded xAI, an artificial intelligence company.

Musk has expressed views that have made him a polarizing figure.[7][8][9] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, transphobia[10][11][12] and antisemitic conspiracy theories.[7][13][14][15] His ownership of Twitter has been similarly controversial, including; laying off a large number of employees, an increase in hate speech and misinformation and disinformation on the website, as well as changes to Twitter Blue verification. In 2018, the U.S. Securities and Exchange Commission (SEC) sued him for falsely tweeting that he had secured funding for a private takeover of Tesla. To settle the case, Musk stepped down as the chairman of Tesla and paid a $20 million fine.
"""


def ice_breaker(
    name: str = "", github_account_name: str = ""
) -> Tuple[PersonIntel, str]:
    # For Test url
    temp_data = {
        "chaejin": "https://gist.githubusercontent.com/ChaejinE/81673f47a86e6a0fb0c0a4aa604e0809/raw/9e8dce0912d20c47d94027eb8581d3f9b756c20c/gistfile1.txt",
        "eunji": "https://gist.githubusercontent.com/ChaejinE/af15dc5b4b99c0ff8dec28b916fa085f/raw/2cd3ca8d004c461e17ada65e895a7cecde5bd1c3/gistfile1.txt",
    }
    linkedin_profile_url = temp_data[name]
    linkedin_data = scrap_linkedin_profile(api_endpoint=linkedin_profile_url)

    # For Real Operation url
    # linkedin_profile_url = linkedin_lookup_agent(name=name)
    # linkedin_data = scrap_linkedin_profile_origin(
    #     linkedin_profile_url=linkedin_profile_url
    # )

    summary_template = """
        given the Linkedin infromation {linkedin_infromation} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. A topic that may interest them
        4. 2 creative Ice breakers to open a conversation with them
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_infromation"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    # GPT Model 3.5
    # temperature : 얼마나 creative ?
    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.run(linkedin_infromation=linkedin_data)
    print("result : ", result)
    print("profile picture url : ", linkedin_data["profile_pic_url"])

    return person_intel_parser.parse(result), linkedin_data["profile_pic_url"]


if __name__ == "__main__":
    result = ice_breaker(github_account_name="ChaejinE")
    print("Result: \n", result)
