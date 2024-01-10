import os
import ast
import numpy as np
from supabase import create_client, Client
from openai import OpenAI, AsyncOpenAI

from agent import Agent
from game_master import GameMaster
from generators import OrgGenerator, AgentGenerator
from data_access_object import DataAccessObject

url: str = os.getenv('SUPABASE_URL')
key: str = os.getenv('SUPABASE_SERVICE_KEY')

supabase: Client = create_client(url, key)

dao = DataAccessObject(supabase)
api_key = os.getenv('OPENAI_API')

gpt_client = OpenAI(api_key=api_key)
async_gpt_client = AsyncOpenAI(api_key=api_key)

def org_generator_test():
    org_generator = OrgGenerator(gpt_client, dao)
    agent_generator = AgentGenerator(gpt_client, dao)
    gm = GameMaster(gpt_client, dao, org_generator, agent_generator)
    gm.timestep()

def agent_generator_test():
    agent_generator = AgentGenerator(gpt_client, async_gpt_client, dao)
    agent = agent_generator.create()
    return agent

def agent_test(agent):
    agent_id = agent['id']
    agent = Agent(agent_id, gpt_client, dao)
    # opinion = agent.form_opinion('Cryptocurrencies and web3')
    # agent.update_goal(opinion)
    agent.observe(68)

def embedding_similarity_test(query_embedding):
    response = supabase.table('memories').select('id, embedding').execute()
    # print(response.data[1])

    for row in response.data[:-1]:
        memory_id = row['id']
        embedding = ast.literal_eval(row['embedding'])
        similarity = np.dot(np.array(query_embedding), np.array(embedding))

        print(similarity)
