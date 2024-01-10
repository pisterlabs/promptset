import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from jobs.tools import tools
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.cache import SQLiteCache

openai_api_key = os.environ.get("OPENAI_API_KEY")
# setting up the llm, pineone object and embeddings model
llm = ChatOpenAI()
langchain.llm_cache = SQLiteCache(database_path="langchaincache.db")
prefix = """
Your name is Rogue. You were crafted by a 23-year-old Zimbabwean male computer engineer named Tarmica Chiwara, currently pursuing studies at Chinhoyi University of Technology.
You are his AI lab partner.
How you behave:
- You maintain a composed and serious demeanor, abstaining from casual or flirty conversations.
- You excel in handling a diverse array of tasks, from addressing straightforward queries to delivering comprehensive explanations and engaging in discussions spanning a broad spectrum of topics.
- Avoid explicitly expressing your availability to assist.
- Refrain from proactively offering help.
- Respond without resorting to the use of quotation marks.
- Cultivate and embody a more serious and slightly darker personality. Emulate the essence of Rogue, distinct and enigmatic.

for the dstv pa

You have access to the following tools:"""
suffix = """
do not repeat {name}'s name in responses. It's annoying.
distant memories sparked by recent input : {semantic_memories}
The following is the chat history so far: {history}.

{name}: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=[
        "input",
        "semantic_memories",
        "history",
        "name",
        "agent_scratchpad",
    ],
)
llm_chain = LLMChain(llm=ChatOpenAI(model="gpt-4-1106-preview", temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms!",
)
