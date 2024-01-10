import os

from langchain.chains.llm import LLMChain
from langchain.memory import ReadOnlySharedMemory
from langchain.agents import BaseSingleActionAgent,  Tool,  AgentExecutor
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseOutputParser,
    OutputParserException
)
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic.v1 import Extra, BaseModel, Field
from typing import Any, List, Tuple, Set, Union



# ROUTER_TEMPLATE = '''あなたの仕事はユーザーとあなたとの会話内容を読み、
# 以下の選択候補からその説明を参考にしてユーザーの対応を任せるのに最も適した候補を選び、その名前を回答することです。
# あなたが直接ユーザーへ回答してはいけません。あなたは対応を任せる候補を選ぶだけです。

# << 選択候補 >>
# 名前: 説明
# {destinations}

# << 出力形式の指定 >>
# 選択した候補の名前のみを出力して下さい。
# 注意事項: 出力するのは必ず選択候補として示された候補の名前の一つでなければなりません。
# ただし全ての選択候補が不適切であると判断した場合には "DEFAULT" とすることができます。

# << 回答例 >>
# 「あなたについて教えて下さい。」と言われても返事をしてはいけません。
# 選択候補に適切な候補がないケースですから"DEFAULT"と答えて下さい。
# '''
ROUTER_TEMPLATE = '''Your job is to read the conversation between the user and yourself, and based on the descriptions provided below, select the most suitable candidate to handle the user's response.
You should not directly answer the user; your role is solely to choose the appropriate candidate.

# Choices
Name: Description
{destinations}

# Output Format
Please output only the name of the selected candidate.
Note: The output must always be one of the names listed as choices. However, if you determine that all provided choices are inappropriate, you may use "DEFAULT".

# Example Answer
If asked, 'Tell me about yourself,' you should not respond.
Since there is no appropriate candidate in the choices, answer with "DEFAULT".
'''

ROUTER_PROMPT_SUFFIX = '''<< Output Format Specification >>
I'll reiterate the instructions one last time. Please output only the name of the candidate you have selected.
Note: The output must always be one of the names listed as choices. However, if you determine that all provided choices are inappropriate, you may use "DEFAULT".
'''


class DestinationOutputParser(BaseOutputParser[str]):
    destinations: Set[str]

    class Config:
        extra = Extra.allow

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.destinations_and_default = list(self.destinations) + ["DEFAULT"]

    def parse(self, text: str) -> str:
        matched = [int(d in text) for d in self.destinations_and_default]
        if sum(matched) != 1:
            raise OutputParserException(
                f"DestinationOutputParser expected output value includes "
                f"one(and only one) of {self.destinations_and_default}. "
                f"Received {text}."
            )

        return self.destinations_and_default[matched.index(1)]

    @property
    def _type(self) -> str:
        return "destination_output_parser"


class DispatcherAgent(BaseSingleActionAgent):

    chat_model: BaseChatModel
    readonly_memory: ReadOnlySharedMemory
    tools: List[Tool]
    verbose: bool = False

    class Config:
        extra = Extra.allow

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        destinations = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        router_template = ROUTER_TEMPLATE.format(destinations=destinations)
        router_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                template=router_template),
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate(prompt=PromptTemplate(
                input_variables=['input'], template='{input}')),
            SystemMessagePromptTemplate.from_template(
                template=ROUTER_PROMPT_SUFFIX)
        ])
        self.router_chain = LLMChain(
            llm=self.chat_model,
            prompt=router_prompt_template,
            memory=self.readonly_memory,
            verbose=self.verbose
        )

        self.route_parser = DestinationOutputParser(
            destinations=set([tool.name for tool in self.tools])
        )

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:

        router_output = self.router_chain.run(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException as ope:
            destination = "DEFAULT"

        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:

        router_output = await self.router_chain.arun(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException as ope:
            destination = "DEFAULT"

        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")


class HoroscopeAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the astrologer.")

# class PartsOrderAgentInput(BaseModel):
#     user_utterance: str = Field(
#         description="プラモデルの部品の個別注文の担当者に伝達するユーザーの直近の発話内容です。")


class SearchDBAgentInput(BaseModel):
    user_utterance: str = Field(
        description="The user's most recent utterance that is communicated to the person in charge of the school database search.")


class DefaultAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that communicates general content to the person in charge.")


# tools = [
#     Tool.from_function(
#         func=horoscope.run,
#         name="horoscope",
#         description="星占いの担当者です。星占いに関係する会話の対応はこの担当者に任せるべきです。",
#         args_schema=HoroscopeAgentInput,
#         return_direct=True
#     ),
#     # Tool.from_function(
#     #     func=parts_order_agent,
#     #     name="parts_order_agent",
#     #     description="プラモデルの部品の個別注文の担当者です。プラモデルの部品注文やキャンセルに関係する会話の対応はこの担当者に任せるべきです。",
#     #     args_schema=PartsOrderAgentInput,
#     #     return_direct=True
#     # ),
#     Tool.from_function(
#         func=search_database_agent.run,
#         name="searchDB",
#         description="学校データベース検索の担当者です。学校データベースの検索や学校情報に関係する会話の対応はこの担当者に任せるべきです。",
#         args_schema=SearchDBAgentInput,
#         return_direct=True
#     ),
#     Tool.from_function(
#         func=default_agent.run,
#         name="DEFAULT",
#         description="一般的な会話の担当者です。一般的で特定の専門家に任せるべきでない会話の対応はこの担当者に任せるべきです。",
#         args_schema=DefaultAgentInput,
#         return_direct=True
#     ),
# ]

# dispatcher_agent = main.DispatcherAgent(
#     chat_model=llm, readonly_memory=readonly_memory, tools=main.tools, verbose=verbose)

# agent = AgentExecutor.from_agent_and_tools(
#     agent=dispatcher_agent, tools=main.tools, memory=memory, verbose=verbose
# )

# def run(input: str):
#     return agent.run(input)