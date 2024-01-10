import os

from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents.react.base import DocstoreExplorer

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


if __name__ == '__main__':
    docstore = DocstoreExplorer(Wikipedia())

    tools = [
        Tool(
            name='Search',
            func=docstore.search,
            description="useful for when you need to ask with search",
        ),
        Tool(
            name='Lookup',
            func=docstore.lookup,
            description="useful for when you need to ask with lookup",
        ),
    ]

    llm = OpenAI(temperature=0, model_name='text-davinci-002')
    react = initialize_agent(tools=tools,
                             llm=llm,
                             agent=AgentType.REACT_DOCSTORE,
                             verbose=True)

    question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the " \
               "United Kingdom under which President?"

    react.run(question)

