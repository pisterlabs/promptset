import openai

from builtEnvironment import BuiltEnvironment
from agent import Agent
from utils import save_json_to_file, parse_json_from_response_text
from llm.base import ChatSequence, Message
from llm.chat import chat_with_gpt
from llm.base import init_openai_key


design_goal_format = \
    "You should only respond in JSON format as described below: " \
    'Response Format: \n{"goals": [\n' \
    '{"title": "","content": "", }\n' \
    '{"title": "","content": "", }\n' \
    '{"title": "","content": "", }\n' \
    "]}" \
    "\nEnsure the response can be parsed by Python json.loads"


def prompt_for_design_goal(agent: Agent, environment: BuiltEnvironment):
    prompt = "You are a design assistant that propose three most important design goals for a house architectural design. "
    prompt += f"The dweller of the house is called {agent.name}. "
    prompt += agent.describe_character()
    prompt += "Please propose three most important design goals for the house. Keep the title short but the content long. "
    prompt += design_goal_format

    return prompt


def get_design_goal_json(agent: Agent, environment: BuiltEnvironment):
    # generate action plan
    init_prompt = prompt_for_design_goal(agent, environment)
    print(init_prompt)

    chat_seq = ChatSequence()
    chat_seq.append(Message(role="system",
                            content="You are a program that generates character action lists for game animation."))
    chat_seq.append(Message(role="user", content=init_prompt))
    result_str, chat_seq = chat_with_gpt(chat_seq)
    print(result_str)

    json_obj = parse_json_from_response_text(result_str)
    return json_obj


class DesignGoal:
    def __init__(self, title: str, content: str):
        self.title = title
        self.content = content


class DesignGoals:
    goals = None

    def __init__(self, agent: Agent, environment: BuiltEnvironment):
        json_obj = get_design_goal_json(agent, environment)
        self.parse_json(json_obj)

    # for iteration, return a list of design goals
    def __iter__(self):
        return self.goals.__iter__()

    def to_json(self):
        return {
            "goals": [goal.__dict__ for goal in self.goals]
        }

    def parse_json(self, json_obj):
        self.goals = []
        for goal in json_obj['goals']:
            self.goals.append(DesignGoal(goal['title'], goal['content']))
        return self


if __name__ == '__main__':
    init_openai_key()
    agent_action = get_design_goal_json(Agent(), BuiltEnvironment())
    save_json_to_file(agent_action, 'results/goals.json')