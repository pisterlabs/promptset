# Copyright © Microsoft Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from wada.agents import HostAgent
from wada.generators import SystemMessageGenerator
from wada.typing import RoleType
from wada.utils import openai_api_env_required

topic = "Living in Beijing or Chengdu?"
summary = "The person prioritizes a larger number of job options and am willing to compromise on the cost of living for better job prospects or social life. And he prefers a milder, more temperate climate and favor a more relaxed and nature-oriented social environment. Additionally, He is seeking top-tier education and healthcare institutions."
aspects = "1. Job opportunities; 2. Cost of living; 3. Quality of life; 4. Transportation and infrastructure; 5. Climate and environment"
messages = """AI Debater A:

Argument: Chengdu offers a more relaxed and nature-oriented social environment.
Explanation: While it is true that Beijing has a larger job market and more top-tier education and healthcare institutions, the person in question has expressed a preference for a milder climate and a more relaxed, nature-oriented social environment. Chengdu, with its mild and humid climate, leisurely teahouses, and close proximity to natural attractions like the Panda Research Base and Jiuzhaigou, is better suited to meet these preferences. Additionally, Chengdu's growing tech hub and improving education and healthcare facilities still provide ample opportunities for personal and professional growth. Although Beijing's advantages in job prospects and social life are undeniable, it is essential to consider the person's priorities and preferences, which Chengdu can better cater to in terms of climate and lifestyle.

AI Debater B:

Argument: Beijing's international atmosphere and larger expat community offer a more diverse social life.
Explanation: While Chengdu does provide a more relaxed and nature-oriented social environment, it is important to consider the person's preference for a larger number of job options and better social life. Beijing's international atmosphere and larger expat community offer a more diverse social life, which can be beneficial for personal and professional networking. Furthermore, the higher number of job opportunities in Beijing, particularly in the technology and finance sectors, aligns with the person's willingness to compromise on the cost of living for better job prospects. Although Chengdu's climate and lifestyle may be more appealing, the advantages of living in Beijing in terms of job opportunities and social life should not be overlooked, as they are also important factors in the person's decision-making process.
"""


@pytest.fixture
def host_agent() -> HostAgent:
    host_sys_msg = SystemMessageGenerator().from_dict(
        dict(topic=topic, summary=summary, aspects=aspects),
        role_tuple=("Host", RoleType.HOST),
    )
    return HostAgent(host_sys_msg, role_name="Host")


@openai_api_env_required
def test_host_agent(host_agent: HostAgent):
    result = host_agent.step(messages=messages)
    assert "<<<CONTINUE>>>" or "<<<END>>>" in result
