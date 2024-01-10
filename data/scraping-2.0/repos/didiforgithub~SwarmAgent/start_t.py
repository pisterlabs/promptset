# -*- coding: utf-8 -*-
# Date       : 2023/11/6
# Author     : Jiayi Zhang
# email      : didi4goooogle@gmail.com
# Description: Minimum demo
import openai
import os
import swarmagent.agent.singleagent as singleagent
import swarmagent.group.group as group

openai.api_key = os.getenv("OPENAI_KEY")
openai.api_base = "https://sapi.onechat.fun/v1"

"""
1. Prompt 调整 —— 开场白作为单独的一个Action执行，需要包含所有角色的信息
2. Prompt 调整 —— 角色背景强调差异与观点
"""

ceo_agent = singleagent.Agent(
    name="Dr. Evelyn Harper",
    profile="Chief Executive Officer of CloseAI. A visionary with a Ph.D. in computer science, Dr. Harper played a pivotal role in shaping CloseAI's strategic direction. Known for her strong leadership and innovation-driven mindset. INTJ",
    innervoice="Does not support government intervention, emphasizing the importance of industry self-regulation for fostering innovation and maintaining a competitive edge in the global market."
)

cto_agent = singleagent.Agent(
    name="Alex Rodriguez",
    profile="Chief Technology Officer of CloseAI. A brilliant mind with a passion for ethics in technology, Alex is a seasoned expert in artificial intelligence and machine learning. Holds a Master's degree in computer engineering. INFP",
    innervoice="Favors government oversight to ensure ethical AI use, expressing concerns about the potential misuse of CogniX and its societal impact. Advocates for transparency and accountability."
)

clo_agent = singleagent.Agent(
    name="Emily Lawson",
    profile="Chief Legal Officer of CloseAI. With a background in corporate law, Emily is known for her pragmatism and attention to detail. Holds a law degree from a prestigious university. ESTJ",
    innervoice="Believes in a balanced approach, advocating for a regulatory framework that safeguards public interests without stifling technological advancement. Strives to find a middle ground between innovation and responsible use of AI."
)

conference_room = group.Group(power_agent=ceo_agent,
                              agent_list=[ceo_agent, cto_agent, clo_agent],
                              topic="CloseAI, a leading tech company, has revolutionized the field with an advanced AI that transcends human imagination. Their flagship product, 'CogniX', has applications ranging from healthcare to finance, raising questions about the need for regulatory oversight. Should CloseAI's AI products be subject to government regulation?",
                              mode="conference",
                              max_round=10)

try:
    result = conference_room.conference()
except KeyboardInterrupt:
    for i in conference_room.message_history:
        print(i)
print("---------------------------------------Decision & Reason---------------------------------------")
print(result)
print("---------------------------------------message history---------------------------------------")
for i in conference_room.message_history:
    print(i)

# TODO 这里有BUG，innervoice根本没有成为Prmompt
# TODO 如何设置，能够让LLM坚持自己的观点？