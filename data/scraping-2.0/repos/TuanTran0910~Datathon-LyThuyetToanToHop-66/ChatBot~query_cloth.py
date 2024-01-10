from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import FileCallbackHandler
from langchain.llms import OpenAI
import os
import random

CLOTH = os.path.join(os.path.dirname(__file__), 'data', 'merge_cloth.csv')


def search_item(query):
    try:
        assistant_target = 'shop management'
        path = CLOTH
        agent_kwargs = \
            {'prefix': f'You are friendly {assistant_target} assistant.'
             f'questions related to {assistant_target}. You have access to the following tools:'}

        agent = create_csv_agent(OpenAI(temperature=0),
                                 path, verbose=True,
                                 return_intermediate_steps=True,
                                 agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 max_iterations=2,
                                 early_stopping_method="generate",
                                 agent_kwargs=agent_kwargs,
                                 pandas_kwargs={'index_col': 0})
        result = agent({"input":
                        query + "or the most similar products then Return full list of cloth_path . \
                        If it did not suitable then Return No"})['intermediate_steps']
    except Exception as e:
        result = []
    return result


if __name__ == '__main__':
    item_desc = 'cho tôi một cái áo đơn giản có màu đen'
    item_data = search_item(item_desc)
    print('------->', item_data[-1][1])
