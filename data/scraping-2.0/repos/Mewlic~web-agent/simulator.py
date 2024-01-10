import logging

logging.basicConfig(level=logging.ERROR)
import argparse
from yacs.config import CfgNode
from tqdm import tqdm
import os
import json
import faiss

from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.experimental.generative_agents import (
    GenerativeAgentMemory,
)
import math
from agents.data import Data
from agents.recagent import RecAgent
from agents.task import Task
from agents.task2 import Task2

class Simulator:
    """
    Simulator class for running the simulation.
    """
    def __init__(self, config: CfgNode):
        self.config = config
        os.environ["OPENAI_API_KEY"] = self.config["api_keys"][0]
        self.data = Data(self.config)
        self.agents = self.agent_creation()

    def relevance_score_fn(self, score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        return 1.0 - score / math.sqrt(2)

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, other_score_keys=["importance"], k=15
        )

    def generate_step(self, data_path, actions_old=None):
        agent = self.agents[0]
        actions_allusers = actions_old if actions_old else dict()

        for user_id in agent.data.users:
            if user_id in actions_allusers or user_id in ['1701110210']:
                continue
            actions_alltasks = dict()
            for task_id in agent.data.users[user_id]:
                if task_id not in ['1', '4', '7', '8', '10']:
                    continue
                actions_onetask = []
                task = Task2(task_id, self.data.prompt, self.data.background, agent, self.config['strategy'],
                            self.config['mode'], actions_onetask, agent.data.users[user_id][task_id]['content'])
                step = 0
                end = 0
                while not end:
                    actions_onetask, end = task.generate_step(step, self.config['max_result_token'],
                                                              self.config['max_content_token'])
                    step += 1
                actions_alltasks[task_id] = actions_onetask
            actions_allusers[user_id] = actions_alltasks

            with open(data_path, "w", encoding='utf-8') as file:
                json.dump(actions_allusers, file, default=lambda o: o.__dict__, indent=4, ensure_ascii=False)

    def generate_session(self, actions_old=None):
        agent = self.agents[0]
        actions_all = actions_old if actions_old else dict()

        task_id = '3'

        actions = []
        task = Task2(task_id, self.data.prompt, self.data.background, agent, self.config['strategy'],
                    self.config['mode'], actions)
        step = 0
        end = 0
        while (not end) and (step < 16):
            actions, end = task.generate_session(step, self.config['max_result_token'], self.config['max_content_token'])
            step += 1

        actions_all[task_id] = actions
        return actions_all

    def create_agent(self, i, api_key):
        """
        Create an agent with the given id.
        """
        LLM = ChatOpenAI(max_tokens=self.config['max_token'], temperature=self.config['temperature'], openai_api_key=api_key)
        agent_memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=8
        )
        agent = RecAgent(
            id=i,
            data=self.data,
            history=[],
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            config=self.config,
            memory=agent_memory,
        )
        return agent

    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        api_keys = list(self.config['api_keys'])
        num_agents = int(self.config['num_agents'])

        for i in tqdm(range(num_agents)):
            api_key = api_keys[i % len(api_keys)]
            agent = self.create_agent(i, api_key)
            agents[agent.id] = agent
        return agents

    def run(self):
        folder_path = os.path.join('output', self.config['mode'], self.config['strategy'])
        os.makedirs(folder_path, exist_ok=True)

        if self.config['mode'] == 'generate_step':
            data_path = os.path.join(folder_path, 'actions.json')
            if os.path.exists(data_path):
                with open(data_path, "r", encoding='utf-8') as file:
                    actions_old = json.load(file)
                self.generate_step(data_path, actions_old)
            else:
                self.generate_step(data_path)

        elif self.config['mode'] == 'generate_session':
            data_path = os.path.join(folder_path, 'actions.json')
            if os.path.exists(data_path):
                with open(data_path, "r", encoding='utf-8') as file:
                    actions_old = json.load(file)
                actions = self.generate_session(actions_old)
            else:
                actions = self.generate_session()

            with open(os.path.join(folder_path, 'actions.json'), "w", encoding='utf-8') as file:
                json.dump(actions, file, default=lambda o: o.__dict__, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default='config/config.yaml', help="Path to config file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)

    optional_mode = ['generate_step', 'generate_session']
    optional_strategy = ['direct-example', 'direct-summary', 'reasoning-example', 'reasoning-summary',
                         'multi_level-example', 'multi_level-summary', 'reasoning-multi_level-example',
                         'reasoning-multi_level-summary', 'direct-none', 'reasoning-none']

    if config['mode'] not in optional_mode or config['strategy'] not in optional_strategy:
        print('策略错误')
        return

    # run
    recagent = Simulator(config)
    recagent.run()


if __name__ == "__main__":
    main()
