import openai

from builtEnvironment import BuiltEnvironment
from designGoal import DesignGoal, DesignGoals
from agent import Agent
from llm.base import ChatSequence, Message
from llm.chat import chat_with_gpt
from llm.base import init_openai_key


design_goal_format = \
    "You should only respond in JSON format as described below: " \
    'Response Format: \n{"reflections": [\n' \
    '{"related": "yes","content": "describe how the scenario is related to the goal. ' \
    'And how the environment design can be improved in this scenario to better achieve the goal.", }\n' \
    "]}" \
    "\nEnsure the response can be parsed by Python json.loads"

design_goal_format2 = \
    "You should only respond in JSON format as described below: " \
    'Response Format: \n{"reflections": [\n' \
    '{"related": "no","content": "", }\n' \
    "]}" \
    "\nEnsure the response can be parsed by Python json.loads. "


def prompt_for_action_goal_reflect(agent: Agent, environment: BuiltEnvironment, goal: DesignGoal):
    prompt = "You are an architectural design assistant " \
             "that reflect on environment design for a given scenario and a given goal. "
    prompt += f"The scenario consists of a character, character's current action, and his current location. "
    prompt += agent.describe_character()
    prompt += f"The current action: yawn. The current location is bed. "
    prompt += f"Please answer whether the action 'yawn' is related to the goal '{goal.title}'. "
    prompt += "If yes, " + design_goal_format
    prompt += "If no, " + design_goal_format2

    return prompt


def action_goal_reflect(agent: Agent, environment: BuiltEnvironment, design_goals: DesignGoals):

    for goal in design_goals:
        init_prompt = prompt_for_action_goal_reflect(agent, environment, goal)
        print(init_prompt)

        chat_seq = ChatSequence()
        chat_seq.append(Message(role="system",
                                content="You are a program that generates character action lists for game animation."))
        chat_seq.append(Message(role="user", content=init_prompt))
        result_str, chat_seq = chat_with_gpt(chat_seq)
        print(result_str)


def main():
    init_openai_key()
    agent = Agent()
    environment = BuiltEnvironment()
    design_goals = DesignGoals(agent, environment)
    action_goal_reflect(agent, environment, design_goals)


if __name__ == '__main__':
    main()
