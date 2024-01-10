from langchain.chat_models import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.schema.agent import AgentFinish
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents.format_scratchpad import format_to_openai_functions
from prompts import *

# Tools
from Tools.search import get_serpapi_search
from Tools.temperature import get_current_temperature
from Tools.mbti import get_mbti_explaination
from Tools.latest import LatestmovieTool

# from typing import Optional, Dict, Any
# from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
# class CustomAgentExecutor(AgentExecutor):
#     def _return(
#         self,
#         output: AgentFinish,
#         intermediate_steps: list,
#         run_manager: Optional[CallbackManagerForChainRun] = None,
#     ) -> Dict[str, Any]:
#         if run_manager:
#             run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
#         final_output = output.return_values
#         if self.return_intermediate_steps:
#             final_output = {"output": final_output, "intermediate_steps": intermediate_steps}
#         return final_output

#     async def _areturn(
#         self,
#         output: AgentFinish,
#         intermediate_steps: list,
#         run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
#     ) -> Dict[str, Any]:
#         if run_manager:
#             await run_manager.on_agent_finish(
#                 output, color="green", verbose=self.verbose
#             )
#         final_output = output.return_values
#         if self.return_intermediate_steps:
#             final_output = {"output": final_output, "intermediate_steps": intermediate_steps}
#         return final_output
    
class ChatAgent():
    def __init__(self, tools, **params):
        super(ChatAgent, self).__init__( **params)
        self.functions = [format_tool_to_openai_function(f) for f in [get_current_temperature, get_serpapi_search, get_mbti_explaination]]
        self.model = ChatOpenAI(temperature=0.4, max_tokens=1024).bind(functions=self.functions)
        self.memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt1_system),
            MessagesPlaceholder(variable_name='chat_history'),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])
        self.chain = RunnablePassthrough.assign(
        agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
         ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        self.qa = AgentExecutor(agent=self.chain, tools=tools, memory=self.memory, verbose=True)
    
    def convchain(self, query):
        if not query:
            return
        result = self.qa.invoke({"input": query, "mbti_list": mbti_list, "recommend_format":recommend_format})
        return result