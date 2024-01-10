from web3_config import w3
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from tools import ERC20Tool, ENSToOwnerAddressTool, EtherscanABIQuery, ExecuteReadTool
from langchain.agents import load_tools
from tools_manager import prepare_tools, select_tools
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from memory import Memory

from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

from typing import Callable


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def get_tools(tools, retriever, query):
    docs = retriever.get_relevant_documents(query)
    return [tools[d.metadata["index"]] for d in docs]


if __name__ == "__main__":
    print(f"Connected to web3: {w3.is_connected()}")

    search = SerpAPIWrapper()
    langchain_tools = [
        ENSToOwnerAddressTool(),
        ERC20Tool(),
        EtherscanABIQuery(),
        ExecuteReadTool(),
        Tool(
            name="web_search",
            func=search.run,
            description=(
                "Useful for when you need to answer questions about current events or the current state of the world. "
                "The input to this should be a single search term. We should also prioritize the other tools if we can "
                "require precise blockchain information."
            )
        )
    ]

    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(langchain_tools)
    ]
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    get_tools_unary = lambda query: get_tools(
        langchain_tools, retriever, query
    )

    langchain_agent = initialize_agent(
        langchain_tools,
        chat,
        max_iterations=1,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=conversation_memory,
        verbose=True
    )

    # tools_with_embeddings = prepare_tools(langchain_tools)
    # simple_memory = Memory()
    # for tool in langchain_tools:
    #     simple_memory.update_memory(
    #         f"We can use the {tool.name} tool to: {tool.description}"
    #     )
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""
    prompt = CustomPromptTemplate(
        template=template,
        tools_getter=get_tools_unary,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()
    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in langchain_tools]
    agent_tool_suggest = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
        verbose=True
    )

    try:
        while True:
            print("BOT: Input a query!")
            query = input()



            relevant_tools = get_tools_unary(query)[:3]
            print("BOT: Relevant tools to use:")
            for i, tool in enumerate(relevant_tools):
                print(i, tool.name)
                print("---> ", agent_tool_suggest(""))

            print("Write 0-2, or no")

            inp = input()
            if inp == "no":
                continue
            else:
                try:
                    idx = int(inp)

                except Exception as e:
                    str(e)
                    continue
            # langchain_agent.run(query)
            # print(f"Langchain suggestion: {langchain_agent.}")

            # simple_memory.update_memory(
            #     f"User input was #{query}#, and the smart bot told us: #{langchain_output}#."
            # )

    except Exception as e:
        print(e)