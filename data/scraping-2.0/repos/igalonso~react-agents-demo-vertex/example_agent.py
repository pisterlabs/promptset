from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import VertexAI
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
import wikipedia
import vertexai

MODEL_NAME = "text-bison@001"

# This is the langchain connection to Vertex AI.
# Note this depends on vertexai.init (which was run in Part 0).
llm = VertexAI(model_name=MODEL_NAME, temperature=0)
verbose=True
# Initialize the Wikipedia tool.
_ = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# This next line invisibly maps to the previous line. The WikipediaQueryRun
#   call is what matters here for Langchain to use its "wikipedia", not
#   the variable that call is output to.
tools = load_tools(["wikipedia"], llm=llm)

# Create the ReAct agent.
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=verbose)

# You can change this question to see how the agent performs.
# You may get a GuessedAtParserWarning from the wikipedia API, ignore it.
print(agent.run("What US President costarred with a chimp in 'Bedtime for Bonzo'?"))
#print(agent.run("Who was the writer of the book that has a character called Captain Walton in it and features a monster that becomes alive by mad scientist?"))
#print(agent.run("What was the first movie that reached 100 million dollars and was directed by Nacho Vigalondo?"))


# Make the llm-math tool available to the agent.
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=verbose)
print(agent.run("What's 67^2?"))