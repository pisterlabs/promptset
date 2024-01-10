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
"""
效果一般，出现大量重复的
"""

ceo_agent = singleagent.Agent(name="Zhang Wei",
                              profile="Zhang Wei, as the CEO, is a forward-thinking leader who believes in harnessing cutting-edge technology to stay ahead of the competition. She has a keen interest in adopting AI to streamline operations and enhance product offerings. Despite the high costs, she is inclined towards investing in Huawei's GPUs as a long-term strategy for AI development.")
cto_agent = singleagent.Agent(name="Li Jie",
                              profile="Li Jie, the CTO, is deeply tech-savvy with a practical approach. He is meticulous about technical specifications and performance metrics. Although he understands the benefits of Huawei's GPUs for AI, he is cautious about compatibility issues and support. He prefers a balanced approach, weighing the pros and cons of the hardware in the context of the company's specific AI training needs.")
cfo_agent = singleagent.Agent(name="Wang Hong",
                              profile="Wang Hong, serving as the CFO, is very cost-conscious and data-driven. She constantly analyzes the financial implications of any investment and is wary of expenditures that do not promise a clear return on investment (ROI). She acknowledges the potential of AI but is advocating for a thorough cost-benefit analysis before approving the purchase of Huawei's GPUs.")

conference_room = group.Group(ceo_agent, [ceo_agent, cto_agent, cfo_agent],
                              "whether to purchase Huawei's GPUs for AI training", mode="conference", max_round=10)

print(conference_room.power_agent)
try:
    result = conference_room.conference()
except KeyboardInterrupt:
    for i in conference_room.message_history:
        print(i)

print(result)
print("---------------------------------------message history---------------------------------------")
for i in conference_room.message_history:
    print(i)
