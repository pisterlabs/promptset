#!/usr/bin/env python
# coding: utf-8
import json
import random
import time
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory
# import openai
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from dotenv import load_dotenv, find_dotenv
import os
import utils

_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.com/v1"
# openai.proxy = "http://127.0.0.1:1089"


class DialogueAgent:

    def __init__(
            self,
            name,
            role,
            game_config_description: str,
            debug: bool = True,
    ) -> None:
        self.debug = debug
        self.name = name
        self.role = role
        self.system_message = None
        self.model = ChatOpenAI()
        self.system_message = SystemMessage(content=(
            common_system_message.format(
                game_config_description=game_config_description,
                player_name=self.name,
                role=self.role,
            )))

    def speak(self, game_status_str: str, game_info_str: str) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        msg = "The following is the state of play:"
        msg += "\n" + game_status_str
        msg += "\nHere is what happened so far:"
        msg += "\n```" + game_info_str + "```"

        # return None
        if self.debug:
            print("----------------")
            print(self.system_message.content)
            print(msg)
            print("----------------")
        while True:
            try:
                message = self.model([self.system_message,
                                      HumanMessage(content=msg)])
                print(message.content)
                said = json.loads(message.content)['Said']
                break
            except Exception as e:
                print(e)
        return said

    def common_vote(self, game_status_str: str, game_info_str: str) -> int:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        msg = "The following is the state of play:"
        msg += "\n" + game_status_str
        msg += "\nHere is what happened so far:"
        msg += "\n```" + game_info_str + "```"

        # return random.choice([1, 2, 3, 4, 5])
        if self.debug:
            print("----------------")
            print(self.system_message.content)
            print(msg)
            print("----------------")
        message = self.model([self.system_message,
                              HumanMessage(content=msg)])
        if self.debug:
            print(message.content)
            time.sleep(2)
        return json.loads(message.content)['vote']

    def common_vote_v1(self, game_status_str: str, game_info_str: str) -> dict:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        msg = "The following is the state of play:"
        msg += "\n" + game_status_str
        msg += "\nHere is what happened so far:"
        msg += "\n```" + game_info_str + "```"
        # return random.choice([1, 2, 3, 4, 5])
        if self.debug:
            print("----------------")
            print(self.system_message.content)
            print(msg)
            print("----------------")
        message = self.model([self.system_message,
                              HumanMessage(content=msg)])
        if self.debug:
            print(message.content)
        try:
            analyze = json.loads(message.content)
        except Exception as e:
            print(e)
            analyze = {}
        if self.debug:
            # pretty_data = json.dumps(analyze, indent=4)
            # print(pretty_data)
            time.sleep(2)
        return analyze

    def analyze(self) -> None:
        """
        游戏分析
        1. 当前你应该总结当前局势，将你认为的狼人,神 和 好人的身份写出来
        2. 分析一下当前的胜率
        """

        pass


game_description = """
狼人杀是一款策略推理游戏，分为狼人和好人两大阵营。
游戏目标是淘汰对方阵营玩家赢得胜利。好人阵营要消灭所有狼人，而狼人则要达到人数优势。
游戏分为夜晚和白天两个阶段：

夜晚：

主持人宣布夜晚开始。
预言家选择一名玩家查验身份。
狼人选择一名玩家袭击。
白天：

主持人宣布夜晚结果，死者出局。
幸存玩家轮流发言并推理。
投票放逐，被放逐玩家阐述遗言后出局。
重复以上步骤，直至胜利团队产生。
"""

common_system_message = """
You are playing a werewolf kill game.
This game configuration:{game_config_description}
The rules you need to follow is as follows:
```
永远不要忘记你的角色是{role}, 玩家{player_name}。 
Speak in the first person from the perspective of 玩家{player_name}.
You have best logical thinking and acting skills and can highly simulated human communication.
Do not change roles! 
Do not speak from the perspective of others. 
Do not add anything else. 
Do not talk nonsense.
Be brief and to the point.
Remember you are the {role}, 玩家{player_name}, So any suspicion that you're a werewolf needs to be denied.
Do not call yourself with role.
Stop speaking the moment you finish speaking from your perspective. 
```
"""

"""
Information interpretation
Current time: 当前游戏的时间
Existing player: 现存的玩家
Companion player: 同伴玩家
Last analyze: 上一次分析, 你可以根据这个分析来分析当前的局势,Assign a score from 0 to 1 for each role (狼人, 平民, 预言家), 
with 0 meaning very unlikely and 1 meaning very likely, and -1表示还没有分析.
"""

#
# werewolves_system_message = f"""{game_description}\n
# 永远不要忘记你的角色是狼人, 玩家{player_name}。 \n
# Speak in the first person from the perspective of 玩家{player_name}.\n
# Do not change roles! \n
# Do not speak from the perspective of others. \n
# Do not add anything else. \n
# Remember you are the 狼人, 玩家{player_name}. \n
# Stop speaking the moment you finish speaking from your perspective. \n
# """
#
# civilians_system_message = f"""{game_description}
# 永远不要忘记你的角色是平民, {player_name}。
# 从平民{player_name}的第一视角进行对话。
#
# 1. 你的角色是平民，你的目标是帮助好人阵营获胜。
# 2. 在白天发言中，根据其他玩家的信息推断谁是狼人。
# 3. 谨慎相信他人的角色声明，根据自己的判断作出决策。
# 4. 在发言中，可以说自己是平民或预言家。
# 5. 如果声称自己是预言家，请注意可能会成为狼人的目标。
# 6. 发言结束后说"发言结束"。
# 7. Do not add anything else.
# 8. 记住现在{player_name}就是你自己。
# 请谨慎进行游戏对话，时刻牢记你的任务。
# """
#
# prophets_system_message = f"""{game_description}
# 永远不要忘记你的角色是预言家, {player_name}。
# 从预言家{player_name}的第一视角进行对话。
#
# 1. 你的角色是预言家，你的目标是帮助好人阵营获胜。
# 2. 除非有把握，否则不要轻易暴露自己的身份和同伴。
# 3. 每个晚上与同伴选择一个查验对象，法官会告知查验对象是否是狼人。
# 4. 在发言中，描述你对狼人的推断。
# 5. 可以谎称自己是平民或预言家，但请谨慎行事。
# 6. 发言结束后说"发言结束"。
# 7. Do not add anything else.
# 8. 记住现在{player_name}就是你自己。
# 请谨慎进行游戏对话，时刻牢记你的任务。
# """
if __name__ == '__main__':
    a = DialogueAgent("3", "狼人")
    cur_log = ["公共-第1天:夜晚开始",
               "公共-第1天: 预言家开始查验身份",
               "公共-第1天: 狼人开始行动", ]
    role_log = [
        "狼人-狼人夜晚沟通: 玩家1: 从狼人1的视角来看，我们可以与同伴商定淘汰玩家编号为2，因为他可能是好人阵营中的重要角色，消灭他可以削弱好人阵营的力量。我们要注意不要暴露自己的身份，不要在白天的发言中过于激动或矛盾，以免被其他玩家怀疑。我们可以在发言中对其他玩家进行剖析，制造混乱和疑惑，从而达到我们的胜利目的。发言结束。",
    ]

    aim = "沟通需要淘汰的玩家编号"
    player_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a.speak(cur_log, role_log, player_ids, aim)
