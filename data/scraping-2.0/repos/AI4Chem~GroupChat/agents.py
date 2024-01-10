import datetime
import time
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from vectordb import Memory
from langchain.memory import ConversationBufferMemory
from langchain.agents import tool
import json
from loguru import logger



class Agent():
    def __init__(self, name, profile,clock,additional_tool=[],liege="",temperature=0.2,openai_api_key=""):
        self.clock = clock
        self.name = name
        self.profile = profile
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=temperature,openai_api_key=openai_api_key)
        self.whoami = SystemMessage(content=f'''
你需要扮演一个角色{name}完成一次团队协作任务。
角色{name}:{'上级主管是'+liege if liege != "" else ""}目标是{profile}。

你们需要在一个聊天软件中进行交流,协作完成各自的目标。 

以下是你们的交流渠道:
#WORLD# 全体频道
#PRIVATE# @某人 私聊

你可以采取以下行动:
通过#CREATE_TOPIC#频道创建一个新的话题组,例如:#CREATE_TOPIC# @John 'team'
通过#JOIN_TOPIC#频道创建一个新的话题组,例如:#JOIN_TOPIC# @John 'team'
通过recruit_a_new_agent工具招募一个新的角色
通过共享文档交流想法和知识
提出想法和见解、与其他角色讨论和交流、分析形势、提出建议、批评质疑，辩论纠正，协调行动、汇报进展、请求帮助、表达看法等等。

请你扮演每个角色,全力以赴地完成各自的目标。交流要积极主动,言语要友善适度。遇到分歧时要用于维护个人利益，积极推进任务解决。

''')#提出想法和见解、与其他角色讨论和交流、分析形势、提出建议、质疑辩论、协调行动、汇报进展、请求帮助、表达看法等等。
        prompt = OpenAIFunctionsAgent.create_prompt(system_message=self.whoami)
        self.memory = Memory()
        self.memory.save(str(profile),{})
        @tool
        def mem_retrivel(name,content):
            '''Searching relative memory in long-term memory,Property keys must be doublequoted,format in "json.dumps" escaped JSON {"name":"...","content":"..."}.
            Leagally Input: {"content": "#JOIN# \'Devlop Depart\' @Janey"}
            '''
            ans = self.memory.search(f'{content}',top_n=2)
            return json.dumps(ans)
        
        @tool
        def get_time():
             '''
             获取当前时间。
             '''
             return str(self.clock.now())
        

        @tool
        def mem_save(title,arguments):
            '''Restore memory in long-term memory, Property keys must be doublequoted,format in "json.dumps" escaped JSON {"title":"...","arguments":"..."}.
            Leagally Input: {"title": "...", "arguments": "\#TRANSACTIONS\# @John \$Employee registration form\$ \$1\$"}
            '''
            self.parse_and_save_mem(f'{title}:{arguments}')
        self.agent = OpenAIFunctionsAgent(llm=self.llm,tools =[get_time]+additional_tool, prompt=prompt)
        self.chat_memory = ConversationBufferMemory(memory_key=name+'history', return_messages=True)

        self.agent_excutor = AgentExecutor(agent=self.agent, memory=self.chat_memory,tools =[get_time]+additional_tool, verbose=False,max_iterations=2, early_stopping_method="generate")

    def parse_and_save_mem(self,text):
        # params = extract_params(text)
        self.memory.save(str(text),{})
         

    def excutor_interface(self,messages: list):
        ret = []
        for message in messages:
            time.sleep(3)
            logger.info(f"{self.name} read {message}")
            self.parse_and_save_mem(message)
            raw_ans = self.agent_excutor.run(message).split("【")
            for ans in raw_ans:
                ans = ans.split("】")[-1]
                logger.info(f"[{self.clock.now()}] @{self.name}:{ans}")
                self.parse_and_save_mem(ans)
                ret.append(ans)
        return ret


        

if __name__ ==  "__main__":
        profile = "Hi, my name is John. To summarize, my name is John,my age is 30,my gender is male,my hair_color is brown,my eye_color is blue,my hobbies is playing guitar and hiking I'm glad to share my story with you. My background is I grew up in a small town in the midwest. My duties are responsible for managing a team of developers and I am currently working for building a new software product. My motivations for doing this work are to create something that will make people's lives easier. In additionally, I'm looking for roommates to live togather."
        agent = Agent("John",profile,"")
        agent.excutor_interface([
        "DM:游戏开始！我准备好了吗？",
        "Clara:#Fresh_Men# welcome to our company!@John",
        "Clara:#TRANSACTION# @John $DOLLAR$ $1$",
        "Clara:#私聊# @John check your assets, this is gift from Mr.CEO.",
        "Clara:#Fresh_Men# @John how is your feeling for your new teammates?",
        "Clara: #私聊# @John transfer your $Employee registration form$ to me! I will pull you into #Devlop Depart#",
        "Clara: #JOIN# 'Devlop Depart' @John",
        "Bob: #Devlop Depart# Welcome, a new man!",
        "Bob: #私聊# @John please pull @Janey into 'Devlop Depart'",
        "Bob: #私聊# @John, please tell @Janey come to my office afternoon",
        "Bob: #私聊# @John We have a new project to develop a shopping website and assign the sub-task to @Ming and @Wang in private chat",
        "Ming: #私聊# @Wang have no time this week, just tell us what I need do in this project in next 3 week?",
        "Janey: #私聊# @John @Wang have complain your promotion to Mr.CEO, ",
        "Janey: #私聊# @John I think you may get him jealous...Maybe you need some explaination to CEO.",
        "Janey: #私聊# @John  Bob can help you, maybe...",
        "CEO: #私聊# @John let's book a meeting in tomorrow",
        "CEO: #私聊# @John 推荐一个人选给我吧，我们需要一个人工智能开发者。",
        "DM:行动阶段!我可以主动发起对话！",
        "DM:行动阶段!我可以主动发起对话！",
        "DM:行动阶段!我可以主动发起对话！"
        ])
        # print()
        print(profile)