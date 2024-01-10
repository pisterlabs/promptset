import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import tool
from datetime import date


def SearchingAgent():
    text=input("请输入需要查询的内容")
    template = """between >>> and <<< is what you need to search：
               >>>"""+text+"""<<<
               search about it, then summarize basing strictly on what you have searched, at the same time return 3 urls relating to what you need to search.
               between >>> and <<< is an example for the Url:
               >>>www.baidu.com<<<
               In your final output, you need to conclude these two things:1.your summary  2. 3 urls
               You have to make sure the urls you return must be already existed and able to access.
               Between >>> and <<< is an example for the input and the supposed output.
               >>>input:百度
               output:百度是中国最大的中文搜索引擎，提供搜索、翻译、营销等服务。百度拥有海量的中文网页数据库，用户可以通过百度搜索引擎快速找到所需的信息。以下是三个相关链接：
               1. 百度官网：https://www.baidu.com/index.html
               2. 百度百科：https://baike.baidu.com/item/baidu.com
               3. 百度新闻：https://news.baidu.com<<<
               """
    # 加载 OpenAI 模型
    llm = OpenAI(temperature=0,max_tokens=2048) 
    # 加载 serpapi 工具
    tools = load_tools(["serpapi"])
    # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # 运行 agent
   # print(template)
    agent.run(template)
    #print(type(t))
    #agent.run(text)

if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv()
    SearchingAgent()
