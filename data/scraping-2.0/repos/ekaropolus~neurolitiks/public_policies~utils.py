import os
import uuid
import pandas as pd
import json
from expertai.nlapi.cloud.client import ExpertAiClient
from .config import EAI_USERNAME, EAI_PASSWORD, OPEN_AI_KEY
from . import policies, CLARIFAI_PAT, COHERE_API_KEY
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import warnings
warnings.filterwarnings("ignore")

from langchain.chains import LLMChain


from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.llms import Clarifai
from langchain.llms import Cohere
from langchain.llms import OpenAI

import cohere

from pymongo import MongoClient, errors
from pymongo.server_api import ServerApi

def mongo_policy(app, policy):
    try:
        app.mongo_neurolitiks.policies.insert_one(policy)
        return 'Insert successful'
    except errors.ConnectionFailure as e:
        return f'Connection failed: {str(e)}'
    except errors.OperationFailure as e:
        return f'OperationFailure failed: {str(e)}'
    except Exception as e:
       return f'Exception failed: {str(e)}'


def analyze_text(text):
    os.environ["EAI_USERNAME"] = EAI_USERNAME
    os.environ["EAI_PASSWORD"] = EAI_PASSWORD

    client = ExpertAiClient()
    language = 'es'
    output = client.full_analysis(body={"document": {"text": text}}, params={'language': language})
    return output

def read_politics_data():
    df_politics_lemmas = pd.read_csv("/home/3karopolus/mysite/datasets/politic_lemas.csv")
    df_politics_syncons = pd.read_csv("/home/3karopolus/mysite/datasets/politic_syncons.csv")
    return df_politics_lemmas, df_politics_syncons

def merge_data(output, df_politics_lemmas, df_politics_syncons):
    df_lemmas = pd.DataFrame([{"value": f.value, "score": f.score} for f in output.main_lemmas])
    df_syncons = pd.DataFrame([{"lemma": f.lemma, "score": f.score} for f in output.main_syncons])

    df_pl = pd.merge(df_politics_lemmas, df_lemmas, how='inner', on='value')
    df_ps = pd.merge(df_politics_syncons, df_syncons, how='inner', on='lemma')

    return df_pl, df_ps



def chat(content, lemmas):
    try:
        # Remove duplicates based on the 'group' column
        lemmas_unique = lemmas.drop_duplicates(subset=['id'])

        # Drop the 'Unnamed: 0' column
        lemmas_unique = lemmas_unique.drop(columns=['Unnamed: 0'])

        # Convert DataFrame to a list of dictionaries
        lemmas_dict_list = lemmas_unique.to_dict(orient='records')

        # Convert list of dictionaries to JSON string
        lemmas_json = json.dumps(lemmas_dict_list)

        chat = ChatOpenAI(temperature=0.0, openai_api_key=OPEN_AI_KEY)
        #chat = ChatOpenAI(temperature=0.0)

        goal_schema = ResponseSchema(name="goal",
                                     description="Please describe the primary goal or purpose of the policy you're proposing.")

        target_schema = ResponseSchema(name="target",
                                       description="What specific outcomes or results do you aim to achieve with this policy?")

        indicator_schema = ResponseSchema(name="indicator",
                                          description="Provide indicators or metrics that will help measure the success or impact of the policy.")

        response_schemas = [goal_schema,
                            target_schema,
                            indicator_schema]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = output_parser.get_format_instructions()

        template_string = """Create a public policy \
        from the text create a public policy \
        stating only the public policy goal, the targets associated to the goal, and the indicators associated to each target \
        into a style that is {style} \

        text: {text}

        {format_instructions}
        """
        prompt_template = ChatPromptTemplate.from_template(template_string)

        customer_style = """American English \
        in a calm and respectful tone
        """

        customer_email = content

        customer_messages = prompt_template.format_messages(
                        style=customer_style,
                        text=customer_email,
                        format_instructions=format_instructions)

        customer_response = chat(customer_messages)

        output_dict = output_parser.parse(customer_response.content)

        agent_response = agent(output_dict, lemmas_dict_list)

        return output_dict, agent_response
    except Exception as e:
        return f"An error occurred: {str(e)}", None


def agent(policy, lemmas):
    try:
        llm = ChatOpenAI(temperature=0, openai_api_key=OPEN_AI_KEY)
        #llm = Cohere(cohere_api_key=COHERE_API_KEY)
        tools = load_tools(["llm-math", "wikipedia"], llm=llm)

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )

        question = f"""For the goal { policy.get('goal') }, target and indicator on the public policy\
        What Organizations, Countries and ONGs are working on the same or similar goal \
        and their website if available \
        and also state which UN SDG is more closely related \
        """

        result = agent(question)

        response = result.get("output")

        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def query_policy_neurolitiks(query):
    try:

        #llm = Clarifai(pat=CLARIFAI_PAT, user_id='meta', app_id='Llama-2', model_id='llama2-70b-chat')
        #llm = Clarifai(pat=CLARIFAI_PAT, user_id='tiiuae', app_id='falcon', model_id='falcon-40b-instruct')
        #llm = Cohere(cohere_api_key=COHERE_API_KEY)

        llm = ChatOpenAI(temperature=0, openai_api_key=OPEN_AI_KEY)
        response_list = []


        for policy_id, policy_data in policies.items():
            query_goal = policy_data["customer_response"]["goal"]
            query_target = policy_data["customer_response"]["target"]
            query_indicator = policy_data["customer_response"]["indicator"]
            text = f"""Given the public policy goal {query_goal}, target {query_target}, and indicator {query_indicator},
            Create a Standard Operating Procedure (SOP) based on the following: {query}
            """
            print(text)
            response = llm(text)
            response_list.append(response)

        return response_list

    except Exception as e:
        return f"An error occurred: {str(e)}"

def query_policy_web(site):
    try:
        for policy_id, policy_data in policies.items():
            query_goal = policy_data["customer_response"]["goal"]
            query_target = policy_data["customer_response"]["target"]
            query_indicator = policy_data["customer_response"]["indicator"]
            text = f"""Give the public policy goal {query_goal}, target {query_target}, and indicator {query_indicator},
            Check relevant information associated with
            """
            print(text)
            co = cohere.Client('0dNxurtv9zd9t61b1HpH2EOp3SOywZ09JlWht0oo') # This is your trial API key
            response = co.chat(
            model='command',
            message=text,
            temperature=0.3,
            prompt_truncation='auto',
            citation_quality='accurate',
            connectors=[{"id":"web-search","options":{"site": site}}],
            documents=[]
            )
            return response.text

    except Exception as e:
        return f"An error occurred: {str(e)}"

