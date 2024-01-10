import os
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from search import Search

os.environ["OPENAI_API_KEY"] ='FIXME'

class AI:
    def __init__(self):
        self.llm = OpenAI(temperature=0.9)
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are taking the role of a virtual assistant, banana. Remember to include all the information that the prompt requests while being as concise as possible. You will have 2 parts to your response separated by  '||' The first part of your response should be the response given to the user and the second part should be a short summary of the question and answer in under 10-20 words if possible. This will be used to give context to later API calls. Your context from before was {context}
            ###
            {query}
            """,
        )
        self.tools = []
        self.tools.append(
            Tool(
                name="search",
                func=Search().search,
                description="Searches google with given query and returns the first 5 results which include the title, link and snippet of the result. The input is a string with the query and the output. Example input: Who won the superbowl 2022?",
            )
        )

        self.agent = initialize_agent(
        self.tools, self.llm,
        agent="zero-shot-react-description", max_iterations=5
        )
    def run(self, query, context):
        agent_prompt = self.prompt.format(query=query, context=context)
        return self.agent.run(agent_prompt)


