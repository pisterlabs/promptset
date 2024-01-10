import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from repolya.autogen.workflow import do_res
from repolya.autogen.util import cost_usage
from autogen import ChatCompletion


ChatCompletion.start_logging(reset_counter=True, compact=False)


_task='''UDP-GlcA的大规模合成'''
re = do_res(_task)
print(f"out: '{re}'")
print(f"cost_usage: {cost_usage(ChatCompletion.logged_history)}")

# import os
# from dotenv import load_dotenv
# load_dotenv(os.path.join(_RePolyA, '.env'), override=True, verbose=True)

# from repolya._const import WORKSPACE_AUTOGEN, AUTOGEN_CONFIG
# from repolya._log import logger_autogen

# from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# import autogen

# from langchain.agents import initialize_agent
# from langchain.chat_models import ChatOpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.summarize import load_summarize_chain
# from langchain import PromptTemplate
# import openai
# from dotenv import load_dotenv

# import json
# import requests
# from bs4 import BeautifulSoup


# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# config_list = config_list_from_json(env_or_file=str(AUTOGEN_CONFIG))
# llm_config = {
#     "config_list": config_list,
#     "seed": 42,
#     "request_timeout": 600
# }

# def search(query):
#     url = "https://google.serper.dev/search"
#     payload = json.dumps({
#         'q': query
#     })
#     headers = {
#         'X-API-KEY': 'e66757b85a72a921ca77f03cd1ac4489a3adb3a0',
#         'Content-Type': 'application/json'
#     }
#     response = requests.request("POST", url, headers=headers, data=payload)
#     return response.json()

# def scrape(url: str):
#     headers = {
#         'Cache-Control': 'no-cache',
#         'Content-Type': 'application/json',
#     }
#     data = {
#         'url': url
#     }
#     data_json = json.dumps(data)
#     response = requests.post(
#         "https://chrome.browserless.io/content?token=0177d884-49c4-499c-bb13-a0dc0ab399bb",
#         data=data_json,
#         headers=headers
#     )
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, "html.parser")
#         text = soup.get_text()
#         print(f"web: {text}")
#         if len(text) > 8000:
#             output = summary(text)
#             return output
#         else:
#           return text
#     else:
#         print(f"HTTP request failed with status code {response.status_code}")

# def summary(content):
#     llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n"],
#         chunk_size=5000,
#         chunk_overlap=500
#     )
#     docs = text_splitter.create_documents([content])
#     map_prompt = """
#     Write a detailed summary of the following text for a research purpose:
#     "{text}"
#     SUMMARY:
#     """
#     map_prompt_template = PromptTemplate(
#         template=map_prompt,
#         input_variables=["text"]
#     )
#     summary_chain = load_summarize_chain(
#         llm=llm,
#         chain_type='map_reduce',
#         map_prompt=map_prompt_template,
#         verbose=True
#     )
#     output = summary_chain.run(input_documents=docs,)
#     return output

# def research(query):
#     llm_config_researcher = {
#         "functions": [
#             {
#                 "name": "search",
#                 "description": "google search for relevant information",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {
#                             "type": "string",
#                             "description": "Google search query",
#                         }
#                     },
#                     "required": ["query"],
#                 },
#             },
#             {
#                 "name": "scrape",
#                 "description": "scraping website content based on url",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "url": {
#                             "type": "string",
#                             "description": "website url to scrape",
#                         }
#                     },
#                     "required": ["url"],
#                 },
#             }
#         ],
#         "config_list": config_list
#     }
#     researcher = AssistantAgent(
#         name="researcher",
#         system_message="Research about a given query, collect as many information as possible, and generate detailed research results with loads of technique details with all reference links attached; Add TERMINATE to the end of the research report;",
#         llm_config=llm_config_researcher,
#     )
#     user_proxy = UserProxyAgent(
#         name="me",
#         code_execution_config={
#             "work_dir": WORKSPACE_AUTOGEN,
#             "last_n_messages": 2
#         },
#         is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
#         human_input_mode="TERMINATE",
#     )
#     user_proxy.register_function(
#         function_map={
#             "search": search,
#             "scrape": scrape,
#         }
#     )
#     user_proxy.initiate_chat(researcher, message=query)
#     user_proxy.stop_reply_at_receive(researcher)
#     user_proxy.send(
#         "Give me the research report that just generated again, return ONLY the report & reference links",
#         researcher
#     )
#     return user_proxy.last_message()["content"]

# result = research("UDP-GlcA大规模生成")
# print(f"{result}")

