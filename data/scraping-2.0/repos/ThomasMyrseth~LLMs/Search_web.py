from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.agents import initialize_agent

"""
need google search bots for qualitative and quantitative data (serpAPI)
wolfram alpha for calculations
? notion for storing data 
"""


class Search_web:
    def __init__(self):
        tool_names = ["serpapi", "wolfram-alpha"]
        self.tools = load_tools(tool_names=tool_names)
        self.llm = OpenAI(temperature=0.5)
        self.agent = initialize_agent(tools=self.tools, llm=self.llm, agent="zero-shot-react-description", verbose=False)
        #run agent by agent.run(query) 

    def search_query(self, query):
        response = self.agent.run(query)
        return response


    