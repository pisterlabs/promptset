from langchain.chat_models import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentFinish
# from utils import execute_commands, gpt
from utils import parse
from functions import tools, execute_commands
import os 

os.environ["OPENAI_API_KEY"]= "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"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = f"ShellGPT-{os.environ['USER']}"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "<YOUR-API-KEY>"  # Update to your API key
# print(os.environ['OPENAI_API_KEY'])
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
# llm.invoke("Create a file and open in vim")
tool_dict = {
    "execute_commands": execute_commands,
    # "gpt": gpt
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but you can't execute commands.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
# print(llm_with_tools)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

def run(inp):
    output = agent.invoke({"input": inp, "intermediate_steps": []})
    if isinstance(output, AgentFinish):
        print(parse(output.return_values["output"]))
        return None
    tool = tool_dict[output.tool]
    observation = tool.run(output.tool_input)
    return observation


# run("Create a file and open in vim")