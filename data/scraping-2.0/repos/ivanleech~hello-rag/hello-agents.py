from langchain.llms import CTransformers
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore import Wikipedia
from langchain.agents.react.base import DocstoreExplorer


llm = CTransformers(
    model="/home/ivanleech/apps/github_new/llm/zephyr-7b-beta.Q4_K_M.gguf",
    model_type="mistral",
    lib="avx2",
)
# llm_math = LLMMathChain(llm=llm)

# # initialize the math tool
# math_tool = Tool(
#     name="Calculator",
#     func=llm_math.run,
#     description="Useful for when you need to answer questions about math.",
# )
# # when giving tools to LLM, we must pass as list of tools
# tools = [math_tool]
tools = load_tools(["llm-math"], llm=llm)

prompt = PromptTemplate(input_variables=["query"], template="{query}")

llm_chain = LLMChain(llm=llm, prompt=prompt)

# initialize the LLM tool
llm_tool = Tool(
    name="Language Model",
    func=llm_chain.run,
    description="use this tool for general purpose queries and logic",
)

tools.append(llm_tool)

# initialize the zero-shot agent
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(name="Search", func=docstore.search, description="search wikipedia"),
    Tool(name="Lookup", func=docstore.lookup, description="lookup a term in wikipedia"),
]

# initialize the docstore agent
docstore_agent = initialize_agent(
    tools, llm, agent="react-docstore", verbose=True, max_iterations=3
)


# zero_shot_agent("what is the capital of Malaysia?")
# zero_shot_agent("1+1+1")
docstore_agent("What were Archimedes' last words?")
