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

import re
import pytest

from wada.agents import TopicAgent
from wada.topic import Topic
from wada.utils import openai_api_env_required


@pytest.fixture
def topic_agent() -> TopicAgent:
    topic = Topic(
        content="Should I take a promotion that requires frequent travel "
        "and long hours, or stay in my current role with good work-life "
        "balance to start a family soon? I'm 35 and want to have kids in "
        "the next few years while furthering my career.")
    return TopicAgent(topic=topic)


@openai_api_env_required
def test_topic_break_down(topic_agent: TopicAgent):
    (pos, neg) = topic_agent.break_down_topic()
    assert isinstance(pos, str)
    assert isinstance(neg, str)


@openai_api_env_required
def test_specify_topic(topic_agent: TopicAgent):
    topic_agent.break_down_topic()
    results = topic_agent.specify_topic()
    pattern = r"\d+\. .*"
    assert re.match(pattern, results)


@openai_api_env_required
def test_abbreviate_topic(topic_agent: TopicAgent):
    topic_abbr = topic_agent.abbreviate_topic()
    assert not len(topic_abbr) > 100
