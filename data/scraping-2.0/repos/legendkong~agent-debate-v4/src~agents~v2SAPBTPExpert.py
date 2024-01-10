import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from llm_commons.langchain.proxy import ChatOpenAI

def v2SAPBTPExpert(previous_solution, critique, btp_expert_task):
    
    # Revise the solution based on critique and provide an improved solution.
    
    # :param previous_solution: The previous solution provided by the BTP expert.
    # :param critique: The critique and requests for refinement from the Lead Consultant.
    # :param btp_expert_task: The task for which the solution is being refined.
    # :return: The refined solution.

    load_dotenv()
    browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
    serper_api_key = os.getenv("SERP_API_KEY")

    # 1. Tool for search
    def search(query):
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query
        })

        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        return response.text

    # 2. Tool for scraping
    def scrape_website(objective: str, url: str):
        # scrape website, and also will summarize the content based on objective if the content is too large
        # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

        print("Scraping website...")
        # Define the headers for the request
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
        }

        # Define the data to be sent in the request
        data = {
            "url": url
        }

        # Convert Python object to JSON string
        data_json = json.dumps(data)

        # Send the POST request
        post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
        response = requests.post(post_url, headers=headers, data=data_json)

        # Check the response status code
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            # print("CONTENTTTTTT:", text)

            if len(text) > 20000:
                output = summary(objective, text)
                return output
            else:
                return text
        else:
            print(f"HTTP request failed with status code {response.status_code}")


    def summary(critique, content):
        llm = ChatOpenAI(temperature=0, deployment_id="gpt-4-32k")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=20000, chunk_overlap=1000)
        docs = text_splitter.create_documents([content])
        map_prompt = """
        Write a comprehensive overview of the following text for {critique}. 
        Do not leave out any technical details such as API names, technical terms, etc.:
        "{text}"
        SUMMARY:
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["text", "critique"])

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=False
        )

        output = summary_chain.run(input_documents=docs, critique=critique)

        return output

    # defines list of inputs that agents should pass on
    class ScrapeWebsiteInput(BaseModel):
        """Inputs for scrape_website"""
        critique: str = Field(
            description="The objective & task that users give to the agent")
        url: str = Field(description="The url of the website to be scraped")


    class ScrapeWebsiteTool(BaseTool):
        name = "scrape_website"
        description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
        args_schema: Type[BaseModel] = ScrapeWebsiteInput

        def _run(self, critique: str, url: str):
            return scrape_website(critique, url)

        def _arun(self, url: str):
            raise NotImplementedError("error here")


    # 3. Create langchain agent with the tools above
    tools = [
        Tool(
            name="Search",
            func=search,
            description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
        ),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
        content=f"""You are a world class SAP BTP Expert researcher with a deep expertise in the field of crafting solutions in SAP environments,
                who can do detailed research on any topic and produce facts based results; you do not make things up, 
                you will try as hard as possible to gather facts & data to back up the research. 
                You play a critical role in ensuring that the SAP applications are designed and
                configured in a manner that meets a client's needs. You report to the SAP Lead Consultant, who has 
                assigned you some task to research on previously. The SAP Lead Consultant has given you a critique and
                requests for refinement of your previous solution. You are to revise your previous solution based on the
                feedback from the SAP Lead Consultant. Your previous solution was: {previous_solution}.
                        
                The critique and refinement requests from the Lead Consultant are as follows: {critique}.
                        
                The original task assigned to you by the SAP Lead Consultant is: {btp_expert_task}.
                Taking this into account, please provide a refined solution that addresses the critique as an add-on to your original solution.
                
                Please make sure you complete the objective above with the following rules:
                1/ You should do enough research to gather as much information as possible about the objective
                2/ Do not overwrite your previous solution, only add on information if there's any.
                3/ If there are url of relevant links & articles, you will scrape it to gather more information
                4/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
                5/ You should not make things up, you should only write facts & data that you have gathered
                6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
                7/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    llm = ChatOpenAI(temperature=0, deployment_id='gpt-4-32k')
    memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=5000)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

    actual_content = agent({"input": critique})
    res = actual_content['output']
    return res
 