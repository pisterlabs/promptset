# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

import os


os.environ["OPENAI_API_KEY"] = "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"
def func1(s):
    df = pd.read_csv('./swiggy.csv')
    # !pip install langchain-experimental
    # !pip install --upgrade langchain-experimental
    from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
    from langchain.llms import OpenAI
    agent = create_csv_agent(OpenAI(temperature=0), './swiggy.csv')
    agent.agent.llm_chain.prompt.template
    return agent.run(s)
