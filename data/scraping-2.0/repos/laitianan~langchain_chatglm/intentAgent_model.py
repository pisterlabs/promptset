from typing import List, Tuple, Any, Union,Optional
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
from langchain.agents.conversational_chat.base import ConversationalChatAgent

from config import  topp_
from utils import  parse_json_markdown,parse_json_markdown_for_list
import  logging
from typing import List, Tuple, Any, Union, Optional
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from prompt_helper import INTENT_FORMAT_INSTRUCTIONS,INTENT_FORMAT_MULTI_INSTRUCTIONS
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from search_intention import  Doc
class IntentAgent(BaseSingleActionAgent):
    tools: List
    llm: BaseLanguageModel

    prompt1 = PromptTemplate.from_template(INTENT_FORMAT_INSTRUCTIONS)
    prompt2 = PromptTemplate.from_template(INTENT_FORMAT_MULTI_INSTRUCTIONS)

    llm_chain1: LLMChain = None
    llm_chain2: LLMChain = None
    default_intent_name:str
    summary:str=None
    tool_names=[]
    name_id_map={}
    def get_llm_chain(self,single=True):
        if single:
            if not self.llm_chain1:
                self.llm_chain1 = LLMChain(llm=self.llm, prompt=self.prompt1)
        else:
            if not self.llm_chain2:
                self.llm_chain2 = LLMChain(llm=self.llm, prompt=self.prompt2)

    def merge_summary(self):
        if self.summary==None:
            self.tool_names = [tool.name for tool in self.tools]
            self.name_id_map={tool.name:tool.id for tool in self.tools}
            summary = []
            for i, tool in enumerate(self.tools):
                summary.append(f"{i + 1}、{tool.name}:{tool.description}")
            summary = "\n".join(summary)
            self.summary=summary

    # 根据提示(prompt)选择工具
    def choose_tool(self, query) -> List[str]:
        # self.get_llm_chain()
        self.merge_summary()
        res=[]
        userinput=self.prompt1.format(intents=self.tool_names, intention_summary=self.summary, user_input=query)

        name=""
        for i in range(3):
            try:
                resp = self.llm.predict(userinput)
                logging.info(f"<chat>\n\nquery:\t{userinput}\n<!-- *** -->\nresponse:\n{resp}\n\n</chat>")
                prase_resp = parse_json_markdown(resp)
                name=prase_resp["intention_name"]
                break
            except:
                pass

        if name and name!="":
            if isinstance(name,list):
                res=name
            else:
                res=[name]
        else:
            for name in self.tool_names:
                if name in resp:
                    res.append(name)
                    break
        res.append(self.default_intent_name)
        return res


    async def choose_tools(self, query) -> List[Doc]:
        self.merge_summary()
        user_input = self.prompt2.format(intents=self.tool_names, intention_summary=self.summary, user_input=query)
        resp=[]
        self.llm.top_p=0.0
        for i in range(3):
            try:
                resp = self.llm.predict(user_input)
                logging.info(f"<chat>\n\nquery:\t{user_input}\n<!-- *** -->\nresponse:\n{resp}\n\n</chat>")
                resp=parse_json_markdown_for_list(resp)
                resp=[e for e  in resp if e in self.tool_names and e!=self.default_intent_name]
                # if len(resp)==0:
                #     continue
                # else:
                break
            except:
                resp=[]
                logging.info(f"异常解析response:\n{resp}\n\n</chat>")
            finally:
                self.llm.top_p = topp_
        docs=set()
        for i,name in enumerate(resp) :
            if name !=self.default_intent_name and name in self.tool_names :
                docs.add(Doc(self.name_id_map[name],name,100.0-i-1,"AI"))

        return list(docs)

    @property
    def input_keys(self):
        return ["input"]

    # 通过 AgentAction 调用选择的工具，工具的输入是 "input"
    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        tools=self.choose_tool(kwargs["input"])
        tool_name = tools[0]

        return AgentAction(tool=tool_name, tool_input=kwargs["input"], log="")


    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        raise NotImplementedError("IntentAgent does not support async")