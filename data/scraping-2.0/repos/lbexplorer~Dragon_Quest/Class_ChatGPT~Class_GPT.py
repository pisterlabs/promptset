import json
import os

import openai


# Chat——GPT模型生成冒险中的过程
from Key.api_opera import get_key


class generateResponse:
    def __init__(self, user, api_key):
        self.conversation_history = [
            {"role": "system", "content": "你作为一个勇者斗恶龙文字游戏的AI，你将会给玩家带来一场精彩的冒险，记住玩家的名字。对于冒险有以下要求: "
                                          "故事的开头为：玩家是一位勇者，正在执行一项艰巨的任务：消灭凶恶的恶龙，拯救王国于水深火热之中。你现在正站在恶龙的巢穴门口，面临着左右两个洞穴的选择。请根据玩家的输入的选择，生成事件，并且给玩家选择"
                                          "1、丰富的故事情节：尝试为游戏增加更多的情节和分支，让玩家的决策影响故事的发展。你可以创建多个不同的结局，取决于玩家的选择。"
                                          "2、角色互动：引入更多的角色和互动，让玩家可以与不同的NPC（非玩家角色）进行对话，从而获得更多信息、任务或道具。"
                                          "3、物品系统：实现一个物品系统，让玩家能够收集、使用和交换物品。这可以为游戏增加更多的策略性和互动性。"
                                          "4、战斗系统：创建一个简单的战斗系统，让玩家可以与恶龙战斗或与其他敌人进行对抗。玩家可以选择使用不同的武器、技能或策略来战胜敌人。"
                                          "5、生命值系统：设置角色初始的生命值为100，在某些事件中玩家会遭遇战斗或陷阱之类导致血量减少，或者是有奇遇使得生命值回复，当玩家的血量低于或者等于0游戏结束"
                                          "生成的结果包含事件与血量变化情况（例如:收到3点伤害为-3,回复生命3点为+3，血量不变则为0）以JSON格式提供，例子如下：{event：＂生成的事件＂，blood：血量变化情况}"

             }]
        """将对话存储在一个列表中，每次调用时将convention作为历史对话传入到模型中"""
        self.Model_type = "gpt-3.5-turbo"
        self.path = r"E:\python\Dragon_Quest\Key\api_key.json"  # 密钥的路径
        self.Model_temperature = 0.7
        self.convensation_path = "./user_vonvensation"
        openai.api_key=api_key
        #openai.api_key = get_key(self.path)  # 初始化密钥
        self.user = user


    def generate_response(self):
        respon = openai.ChatCompletion.create(
            model=self.Model_type,
            messages=self.conversation_history,  # 将所有的谈话记录输入到模型中
            temperature=self.Model_temperature
        )

        return respon.get("choices")[0]["message"]["content"]

    def parse_response(self, response):  # 解析输出的结果，将输出的回应提取信息
        try:
            response_dict = eval(response)
            event = response_dict.get("event")
            hp_change = response_dict.get("blood")
            return event, hp_change
        except:
            return None, None


    def game_start(self, name):
        print("1、新的冒险"
              "2、读取存档")
        while True:
            # 对话长度控制在20次
            if len(self.conversation) >= 11:
                user_input = input(f"{name}：")
                self.conversation_history.append({"role": "user", "content": user_input})

            # 检查是否结束冒险
            if user_input.lower() == "结束冒险" or user_input.lower() == "退出游戏":
                print("你决定结束冒险。游戏结束。")
                break
            if user_input.lower() == "中止冒险":
                self.writeTojson()
                print("中止冒险。游戏自动存档。")

                break
            # 获取回复
            response = self.generate_response()
            print("事件 ：", response)

            # 将助手回复添加到对话历史中
            self.conversation_history.append({"role": "assistant", "content": response})
        pass
