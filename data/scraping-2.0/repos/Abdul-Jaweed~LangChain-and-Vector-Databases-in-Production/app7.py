from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
google_api_key = os.getenv("GOOGLE_API_KEY")

GOOGLE_CSE_ID=google_cse_id
GOOGLE_API_KEY=google_api_key

llm = OpenAI(
    openai_api_key=apikey,
    model=apikey,
    temperature=0
)

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

search = GoogleSearchAPIWrapper()

# Next, we create the tools that our agent will use.

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    
    Tool(
        name="Summarizer",
        func=summarize_chain.run,
        description="useful for summarizing texts"
    )
]

# We are now ready to create our agent that leverages two tools.

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Let’s run the agent with a question about summarizing the latest news about the Mars rover.

response = agent("What`s the latest news about the Mars rover? Then please summarize the results.")
print(response['output'])

