from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from app.myTool import contentTool, chainTool, agentTool
import json

def pp(prompt):
    print(prompt)
    return prompt
rpp = RunnableLambda(pp)

# chat_prompt = ChatPromptTemplate.from_template("{text}")
output_parser = StrOutputParser()

agent = agentTool.agent_executor

# chat_model = ChatOpenAI(model="gpt-4-1106-preview")
chat_chain = (
    { "input": lambda x: json.loads(x) }
    | rpp
    | agent
    | output_parser
)
