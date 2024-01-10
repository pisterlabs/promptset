from typing import Callable, Any

import requests
from bs4 import BeautifulSoup
from agent.common import return_parse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import format_tool_to_openai_function, convert_pydantic_to_openai_function
from agent.source.model import WebSpiderInput, SourceOutput
from base import Base


class AnimationAssistant:
    def __init__(self, base: Base, llm: ChatOpenAI):
        self.__base = base
        self.__llm = llm
        self.__prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个资源查找助手，负责从网上查找资源，需要从content里面解析资源的链接（磁力链接），从ref解析来源网址并返回"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.__tools = [self.__dhmy()]
        self.__functions = [format_tool_to_openai_function(t) for t in self.__tools]
        self.__functions.append(convert_pydantic_to_openai_function(SourceOutput))
        self.__agent = {
                           "input": lambda x: x["input"],
                           "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps'])
                       } | self.__prompt | self.__llm.bind(functions=self.__functions) | return_parse("SourceOutput", {"links": [], "refs": []})
        self.__agent_execute = AgentExecutor(agent=self.__agent, tools=self.__tools, verbose=True)

    def __dhmy_get_detail(self, link) -> str:
        response = requests.get(
            "https://dmhy.anoneko.com{}".format(link),
            proxies=self.__base.proxy,
            headers=self.__base.http_header
        )
        return BeautifulSoup(response.text, 'html.parser').find(id="resource-tabs").get_text()

    def __dhmy(self) -> Callable[[str], Any]:
        @tool("animationSpider1", args_schema=WebSpiderInput)
        def spider(name: str) -> list:
            """从动漫花园中爬取动漫资源"""
            response = requests.get(
                "https://dmhy.anoneko.com/topics/list?keyword={}".format(name),
                proxies=self.__base.proxy,
                headers=self.__base.http_header
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            # 遍历所有的表格
            trs = soup.find(id="topic_list").find("tbody").find_all("tr")
            response = []
            for tr in trs:
                if len(response) == 2:
                    break
                tr.get_text()
                link = tr.select("td:nth-of-type(3)>a")[0]
                tag, href = link.get_text().strip(), link["href"]
                print(tag, href)
                response.append({"name": tag, "content": self.__dhmy_get_detail(href), "ref": href})
            print(response)
            return response
        return spider

    def find(self, content: str) -> dict:
        # 判断资源的类型
        response = self.__agent_execute.invoke({"input": content}, return_only_outputs=True)
        # 把资源类型解析为枚举
        return response
