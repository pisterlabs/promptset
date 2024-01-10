import os
import requests

from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from src.get_keys import GetAzureKeys
from langchain.utilities import BingSearchAPIWrapper
from langchain.tools import BaseTool

from src.get_keys import GetAzureKeys
from src.general_prompts import *
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())  # read local .env file

keys = GetAzureKeys()

# Bing Search question text
SECTION_3_QUESTION_1 = """
Who are the major shareholders of [client_name] company? Provide with:
- their names
- ownership percentages
- their background information.

Summarise of your findings. Provide your references.
"""

SECTION_3_QUESTION_2 = """
Is [client_name] company is part of a larger group structure? If yes, provide:
- key entities within the group and explain its relationship between the entities, including parent companies, subsidaries and affiliates.
- significant transactions or relationships between the [client_name] and related parties.

Summarise of your findings. Provide your references.
"""

SECTION_5_QUESTION_1 = """
What is the industry or sector of the [client_name] company? Provide:
- size of the industry and sector
- growth rate of the industry and sector
- major current trends of the industry and sector
- future trends of the industry and sector

Summarise of your findings. Provide your references.
"""

SECTION_5_QUESTION_2 = """
Who are the major competitors of [client_name]? What are their market shares and key strengths and weaknesses.
"""

SECTION_6_QUESTION_1 = """
Who are the CEO and board of directors/key executives/Board Member of [client_name] company? Provide as many as possible with:
- their names
- their titles
- their relevant experience, qualifications, and achievements

Summarise of your findings. Provide your references.
"""

class CustomBingSearch(BaseTool):
    """Tool for a Bing Search Wrapper"""
    
    name = "@bing"
    description = "useful when the questions includes the term: @bing.\n"

    k: int = 5
    
    def _run(self, query: str) -> str:
        bing = BingSearchAPIWrapper(k=self.k)
        return bing.results(query,num_results=self.k)
            
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This Tool does not support async")


keys = GetAzureKeys()

def get_bing_search_response(question):
    
    tools = [CustomBingSearch(k=5)]

    # set up openai environment - Jay
    llm_proposal = AzureChatOpenAI(deployment_name="gpt-35-16k",temperature=0,max_tokens=2048)

    agent_executor = initialize_agent(tools=tools,
                                    llm=llm_proposal,
                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    agent_kwargs={'prefix':BING_PROMPT_PREFIX},
                                    callback_manager=None,
                                    handle_parsing_errors=True #By Jay
                                    )

    #As LLMs responses are never the same, we do a for loop in case the answer cannot be parsed according to our prompt instructions
    for i in range(2):
        try:
            response = agent_executor.run(question) 
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

    return response

# TODO hard code for testing

client_name = "GOGOX Holding Limited"
section_name = "Shareholders and Group Structure"

def bing_search_for_credit_proposal(client, section_name):
    disclaimer_of_bing_search = False
    if section_name == "Shareholders and Group Structure":
        input_info_str = ["1. Name: Alibaba Group Holding Limited - Ownership Percentage: 23.3% - Background: Alibaba Group Holding Limited is a multinational conglomerate specializing in e-commerce, retail, internet, and technology. It was founded in 1999 by Jack Ma and is headquartered in Hangzhou, China. Alibaba Group operates various online platforms, including Alibaba.com, Taobao, Tmall, and AliExpress. 2. Name: CK Hutchison Holdings Limited - Ownership Percentage: 19.9% - Background: CK Hutchison Holdings Limited is a multinational conglomerate based in Hong Kong. It operates in various industries, including ports and related services, retail, infrastructure, energy, and telecommunications. CK Hutchison Holdings is one of the largest companies listed on the Hong Kong Stock Exchange. 3. Name: Hillhouse Capital Management, Ltd. - Ownership Percentage: 9.9% - Background: Hillhouse Capital Management, Ltd. is an investment management firm based in Asia. It focuses on long-term investments in sectors such as consumer, healthcare, technology, and services. Hillhouse Capital has a strong track record of investing in innovative and high-growth companies. Please note that the ownership percentages mentioned above are based on the available information and may be subject to change."]
        disclaimer_of_bing_search = True
        #input_info_str.append(get_bing_search_response(SECTION_3_QUESTION_1.replace([client_name], client)))
        #input_info_str.append(get_bing_search_response(SECTION_3_QUESTION_2.replace([client_name], client)))

        # Add a Bing replace example if the Bing search cant extract relevent info
        #for text in input_info_str:
        #    if "I need to search" in text or "I should search" in text or "the search results do not provide" in text:
        #        input_info_str = ["I have found some information about the major shareholders of GOGOX Holding Limited. Here are the details: 1. Name: Alibaba Group Holding Limited - Ownership Percentage: 23.3% - Background: Alibaba Group Holding Limited is a multinational conglomerate specializing in e-commerce, retail, internet, and technology. It was founded in 1999 by Jack Ma and is headquartered in Hangzhou, China. Alibaba Group operates various online platforms, including Alibaba.com, Taobao, Tmall, and AliExpress. 2. Name: CK Hutchison Holdings Limited - Ownership Percentage: 19.9% - Background: CK Hutchison Holdings Limited is a multinational conglomerate based in Hong Kong. It operates in various industries, including ports and related services, retail, infrastructure, energy, and telecommunications. CK Hutchison Holdings is one of the largest companies listed on the Hong Kong Stock Exchange. 3. Name: Hillhouse Capital Management, Ltd. - Ownership Percentage: 9.9% - Background: Hillhouse Capital Management, Ltd. is an investment management firm based in Asia. It focuses on long-term investments in sectors such as consumer, healthcare, technology, and services. Hillhouse Capital has a strong track record of investing in innovative and high-growth companies. Please note that the ownership percentages mentioned above are based on the available information and may be subject to change."]

    #elif section_name == "Industry / Section Analysis":
    #    input_info_str.append(get_bing_search_response(SECTION_5_QUESTION_1.replace([client_name], client)))
    #    input_info_str.append(get_bing_search_response(SECTION_5_QUESTION_2.replace([client_name], client)))

    elif section_name == "Management":
        input_info_str = ["CEO and board of directors/key executives/Board Members of GOGOX company:\n\nExecutive Directors:\n  - Chen Xiaohua (陳小華) - Chairman of the Board\n  - He Song (何松) - Co-Chief Executive Officer\n  - Lam Hoi Yuen (林凱源) - Co-Chief Executive Officer\n  - Hu Gang (胡剛)\n\n- Non-executive Directors:\n  - Leung Ming Shu (梁銘樞)\n  - Wang Ye (王也). The company's Board of Directors consists of 12 Directors, including 4 Executive Directors, 4 Non-Executive Directors, and 4 Independent Non-Executive Directors. Unfortunately, I couldn't find more specific information about their relevant experience, qualifications, and achievements. \n\nReferences:\n1. [GOGOX Holdings Limited - Board of Directors](https://www.gogoxholdings.com/en/about_board.php)\n2. [GOGOX CEO and Key Executive Team | Craft.co](https://craft.co/gogox/executives)\n\nPlease note that the information provided is based on the available  sources and may not be exhaustive."]
        disclaimer_of_bing_search = True
        #input_info_str.append(get_bing_search_response(SECTION_6_QUESTION_1.replace([client_name], client)))

        #print(input_info_str)

        # Add a Bing replace example if the Bing search cant extract relevent info
        #for text in input_info_str:
        #    if "I need to search" in text or "I should search" in text or "the search results do not provide" in text:
        #        input_info_str = ["I have found some information about the CEO and board of directors/key executives/Board Members of GOGOX company. Here are the details:\n\nExecutive Directors:\n  - Chen Xiaohua (陳小華) - Chairman of the Board\n  - He Song (何松) - Co-Chief Executive Officer\n  - Lam Hoi Yuen (林凱源) - Co-Chief Executive Officer\n  - Hu Gang (胡剛)\n\n- Non-executive Directors:\n  - Leung Ming Shu (梁銘樞)\n  - Wang Ye (王也). The company's Board of Directors consists of 12 Directors, including 4 Executive Directors, 4 Non-Executive Directors, and 4 Independent Non-Executive Directors. Unfortunately, I couldn't find more specific information about their relevant experience, qualifications, and achievements. \n\nReferences:\n1. [GOGOX Holdings Limited - Board of Directors](https://www.gogoxholdings.com/en/about_board.php)\n2. [GOGOX CEO and Key Executive Team | Craft.co](https://craft.co/gogox/executives)\n\nPlease note that the information provided is based on the available  sources and may not be exhaustive."]
    else:
        input_info_str = []
    return input_info_str, disclaimer_of_bing_search

# SECTION_3_QUESTION_1 = f"""
#     Who are the major shareholders of {client_name}? Provide with:
#     - their names
#     - ownership percentages
#     - their background information.

#     Summarise of your findings. Provide your references.

#     """

# SECTION_3_QUESTION_2 = f"""
#     Is {client_name} is part of a larger group structure? If yes, provide:
#     - key entities within the group and explain its relationship between the entities, including parent companies, subsidaries and affiliates.
#     - significant transactions or relationships between the {client_name} and related parties.

#     Summarise of your findings. Provide your references.

#     """

# SECTION_5_QUESTION_1 = f"""
#     What is the industry or sector of the {client_name}? Provide:
#     - size of the industry and sector
#     - growth rate of the industry and sector
#     - major current trends of the industry and sector
#     - future trends of the industry and sector

#     Summarise of your findings. Provide your references.

#     """

# SECTION_5_QUESTION_2 = f"""
#     Who are the major competitors of {client_name}? What are their market shares and key strengths and weaknesses.

#     """

# SECTION_6_QUESTION_1 = f"""
#     Who are the CEO and Direector/Board Member of {client_name}? Provide as many as possible with:
#     - their names
#     - their titles
#     - their relevant experience, qualifications, and achievements

#     Summarise of your findings. Provide your references.

#     """
