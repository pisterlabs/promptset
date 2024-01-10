from django_pandas.io import read_frame
import pandas as pd
from ..models import InventoryItem
from langchain.agents import create_pandas_dataframe_agent 
from langchain.llms import OpenAI

#retrieving all the items
items = InventoryItem.objects.all()

item_df = read_frame(items)

from langchain.agents import AgentType

agent = create_pandas_dataframe_agent(OpenAI(temperature = 0.2), df = item_df, 
                                      verbose = False, 
                                      return_immediate_steps = False,
                                      AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    )


def ask_about(prompt):
    p = "behave as a data analyst for a small shop use the data provided to give analytics to the business and give output a if you are talking to him"
    p += prompt
    return(agent(p)['output'])

def random_sug():
    prompt = "behave as a data analyst for a small shop use the data provided to give analytics to the business and give output a if you are talking to him"
    prompt += "Give random insights into the data that might be useful to me who is a shop keeper in the future"
    return agent(prompt)['output']

