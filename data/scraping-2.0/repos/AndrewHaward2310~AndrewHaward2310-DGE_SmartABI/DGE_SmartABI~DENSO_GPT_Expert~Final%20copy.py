
from pydantic import BaseModel
from langchain.agents import tool
from answering import Response
from history_retrive import retrive_history
from langchain.chat_models import ChatOpenAI
import openai


@tool
def get_history(query:str)-> str:
 """Search for history of a machine"""
 return retrive_history(query=query)

@tool
def get_instruction(query:str)->str:
    """Use when you want to solve for a problem of a machine"""
    return Response(query=query)

from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "get_instruction": get_instruction, 
            "get_history": get_history,
        }
        return tools[result.tool].run(result.tool_input)

def function_calling(query):
    from langchain.tools.render import format_tool_to_openai_function
    functions = [
        format_tool_to_openai_function(f) for f in [
            get_history, get_instruction
        ]
    ]
    model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0,openai_api_key='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').bind(functions=functions)

    from langchain.prompts import ChatPromptTemplate
    from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are helpful but sassy assistant"),
        ("user", "{input}"),
    ])
    chain = prompt | model | OpenAIFunctionsAgentOutputParser()

    result = chain.invoke({"input": query})
    print(result.tool)

if __name__ =="__main__":
   #function_calling("Làm thế nào để sửa lỗi int3170 trên máy CNC1")
   function_calling("Tôi muốn xem lịch sử máy hút bụi")

