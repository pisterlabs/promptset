
import os
# Langchain imports
from langchain.agents import load_tools, initialize_agent, Tool, AgentType
from langchain.tools import StructuredTool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import List, Union, Optional
from SearchAPI import SearchAPI
from datetime import datetime
import re
from pydantic import BaseModel, Field
from prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX, PREFIX_RESP
from langchain.agents.structured_chat.output_parser import (
    StructuredChatOutputParserWithRetries,
)
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any
from parameters import OPENAI_SK

os.environ["OPENAI_API_KEY"] = OPENAI_SK

uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')

class GetDocumentsArgs(BaseModel):
    question: str = Field(description="A question or statement about a company's SEC financial documents.")
    sentences: Optional[list] = Field(description="A list of related setences to help narrow down the search. REQUIRED argument.")
    # forms: list = Field(description="A list of SEC forms to search for. Include empty list to search all forms.")
    ticker_list: Optional[list] = Field(description="A list of tickers to search for. Empty list will search all tickers.")
    date_start: Optional[str] = Field(description="The start date of the search. Format is YYYY-MM-DD.")
    date_end: Optional[str] = Field(description="The end date of the search. Format is YYYY-MM-DD.")
    max_results: Optional[int] = Field(description="The maximum number of results to return.")
     

tools = [
    StructuredTool(
        name = "GetDocuments",
        func=SearchAPI.get_documents,
        description="Returns specific information from financial documents such as SEC filings, earnings calls, and news reports. Not useful for general questions where a company or sector are not specified.",
        args_schema=GetDocumentsArgs,
    ),
]

class CustomCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> Any:
        response.generations[0][0].text = response.generations[0][0].text.replace("```json", "```")


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    prefix: str
    format_instructions: str
    suffix: str
    # The list of tools available
    tools: List[Tool]
    # add a variable called use_alt
    use_alt: bool = False
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        # Format them in a particular way
        # Create a tools variable from the list of tools provided
        tool_strings = []
        for tool in tools:
            args_schema = re.sub("}", "}}}}", re.sub("{", "{{{{", str(tool.args)))
            # args_schema = str(tool.args)
            tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")
        formatted_tools = "\n".join(tool_strings)
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = self.format_instructions.format(tool_names=tool_names)

        print('self use alt: ', self.use_alt)
        intermediate_steps = kwargs.pop("intermediate_steps")
        if not self.use_alt:
            template = "\n\n".join([self.prefix, formatted_tools, self.format_instructions, self.suffix])
            self.use_alt = True
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
        else:
            template = PREFIX_RESP
            thoughts = "\nObservation: " + intermediate_steps[0][1] + "\nThought:"
            
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        kwargs['tool_names'] = [tool.name for tool in tools]
        kwargs['date'] = datetime.now().strftime("%Y-%m-%d")
        
        # Create a list of tool names for the tools provided
        formatted = template.format(**kwargs)
        formatted = formatted.replace("\'", "")

        # import IPython ; IPython.embed()
        return [HumanMessage(content=formatted)]
    

# Set up an output parser
output_parser = StructuredChatOutputParserWithRetries()

def run_agent(q, context):
    # Model setup
    prompt = CustomPromptTemplate(
        prefix=PREFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        suffix=SUFFIX,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1, top_p=1)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        # We use "Observation" as our stop sequence so it will stop when it receives Tool output
        # If you change your prompt template you'll need to adjust this as well
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    # agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Have model summarize previous messages
    summary = ""
    print("Context: ", context)
    if len(context) > 0:
        llm_context = []
        llm_context.append(HumanMessage(content="Please summarize the following conversation in one to three sentences:"))
        for message in context:
            if message[0] == 'user':
                llm_context.append(HumanMessage(content=message[1]))
            else:
                llm_context.append(SystemMessage(content=message[1]))
        summary = llm(llm_context).content
        q = "Conversation summary: " + summary + "\n" + "New message: " + q
    # Run agent
    output = agent_executor.run(q, callbacks=[CustomCallback()])

    # Parse UUIDs contained in [] tags
    uuids = re.findall(uuid_pattern, output)
    links = SearchAPI.get_references(uuids)
    # Replace UUIDs with links
    output = output.replace(r"[UUID: ", r"[").replace(r"[$UUID: ", r"[").replace(r"[UUID:", r"[")
    output = re.sub(r'\[METADATA: [^\]]+\]', "", output)
  #  for uuid in uuids:
   #     output = output.replace(f"[{uuid}]", "")
    for uuid, link in links.items():
        output = output.replace(uuid, link)
 #   print(output)
    return output, summary


# Run test of agent
if __name__ == "__main__":
    # Test the agent
    # import IPython ; IPython.embed() ; exit(1)
    with get_openai_callback() as cb:
        agent_executor.run("Provide an overview of Restoration Hardware's balance sheets.")
        print("Number of tokens: ", cb.total_tokens)
