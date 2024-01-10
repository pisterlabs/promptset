from langchain.agents import ZeroShotAgent, Tool
from langchain import PromptTemplate, OpenAI, SerpAPIWrapper, LLMChain


def get_todo_chain():
    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    )

    return LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)

def get_tools():
    todo_chain = get_todo_chain()

    search = SerpAPIWrapper()
    return [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="TODO",
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]


def get_chain_prompt():
    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""

    tools = get_tools()

    return ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )