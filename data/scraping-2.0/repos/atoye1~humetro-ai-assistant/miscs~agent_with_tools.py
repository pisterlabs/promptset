from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema.runnable import RunnablePassthrough

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from llm_tools.HumetroWebSearch import HumetroWebSearchTool
from llm_tools.HumetroFare import HumetroFareTool
from llm_tools.GoogleRoutes_legacy import GoogleRouteTool
from llm_tools.prompts import humetro_system_prompt

import os
import openai
import langchain
import warnings

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


langchain.debug = True
warnings.filterwarnings("ignore")
# Load environment variables from .env file


tools = [
    HumetroWebSearchTool(),
    HumetroFareTool(),
    # HumetroWikiSearchTool(),
    GoogleRouteTool(),
    # TrainScheduleTool(),
    # StationDistanceTool(),
]

functions = [format_tool_to_openai_function(t) for t in tools]

llm = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo-16k").bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", humetro_system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

memory = ConversationTokenBufferMemory(
    llm=ChatOpenAI(),
    max_token_limit=4000,
    memory_key="chat_history",
)

chain = prompt | llm | OpenAIFunctionsAgentOutputParser()

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(
        x['intermediate_steps'])
) | chain

agent_executor = AgentExecutor(
    agent=agent_chain,
    tools=tools,
    memory=memory,
    verbose=True,
    show_intermediate_steps=True,
    show_scratchpad=True,
)

# Testing only single turn conversations.
if __name__ == "__main__":
    import datetime
    questions = [
        '어린이 요금과 해운대 해수욕장까지 가는 길을 알려줘',
        '하단역 근처에 큐병원이 있다는데 어떻게 가지?',
        '감천문화마을에 가는 길을 알려줄래?',
        '자갈치시장에 가려햅니다.',
        '거제도가는 버스가 있다던데 몇번인가요?',
    ]
    mass_phrases = [
        "화장실이 어디있나요?",
        "서면 가려면 어디로 타야되나요?",
        "엘리베이터는 어디 있나요?",
        "엘리베이터 언제 고쳐요?",
        "무인민원발급기",
        "주민등록 등본 발급",
        "도서 대출기가 있나요?",
        "전기 휠체어 충전은 어떻게 하나요?",
        "휴대폰을 충전하고 싶어요",
        # 역사 운영 문의
        "몇시까지 영업하나요?",
        "내일 첫차가 언제인가요?",
        "내일 막차가 언제인가요?",
        "물좀 주세요",
        "잠시만 이 물품좀 보관해 주세요",
        # 도시철도 이용 문의
        "정기권은 어떻게 구매하나요?",
        "잘못 탔어요, 반대편으로 넘어가고 싶어요",
        "카드가 안되요",
        "미개표인지",
        "이미 집표된 카드인지",
        "개표 후 못들어가는 카드인지",
        "승차권을 구입하고 싶은데 현금이 없어요",
        "승차권은 현금으로만 구입 가능한가요?",
        "계좌이체로 현금을 받을 수 있나요?",
        # 우대권, 복지카드
        "교통카드 충전하고 싶어요",
        "우대권은 어디서 발급받나요?",
        "우대권 발급받는 기계가 어디 있나요?",
        "우대권 인식이 안됩니다.",
        "복지교통카드를 잃어버렸어요.",
        # LLM이 처리할 수 없는 이례적인 상황(직원 호출 필요)
        "교통카드 인식이 안되요.",
        "물건을 열차내에서 습득했어요",
        "물건을 역사내에서 습득했어요",
        "물건을 잃어버렸어요",
        "물을 마시고 싶어요",
        "엘리베이터가 고장났어요",
        "사람이 쓰러져 있어요",
        "거스름돈으로 바꾸고 싶어요.",
        "휴대폰으로 탔는데 배터리가 다됐어요",
        "구간초과라고 떠요",
        "표를 잃어버렸어요."
    ]

    for q in mass_phrases:
        result = agent_executor.invoke(
            {"input": q})
        timestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('log.txt', 'a') as f:
            f.write("INPUT : " + q)
            f.write("\n")
            f.write(timestring)
            f.write("\n")
            f.write('OUTPUT : ' + str(result['output']))
            f.write("\n\n")
# result = agent_executor.invoke(
#     {"input": "이걸 관리하는 담당자 이름과 전화번호도 알려주세요."})
# print(result)
# agent_executor.invoke(
#     {"input": "각각의 문화행사는 어디서 하나요?"})
# agent_executor.invoke(
#     {"input": "근데 제 이름은 설동헌이에요."})
# agent_executor.invoke(
#     {"input": "제 이름이 뭐라고 그랬죠? 그리고 제가 처음 한 질문이 무엇이었죠?"})
