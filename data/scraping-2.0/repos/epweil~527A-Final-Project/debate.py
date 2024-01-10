##Ethan Weilheimer 
##CSE 527A LLM Agent Paper 
##Code adapted from Cobus Greyling (https://cobusgreyling.medium.com/two-llm-based-autonomous-agents-debate-each-other-e13e0a54429b)
import time
from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field
from langchain.tools import tool
from utils import read_json_file, VIEW_DEBATE, VALID_ACTIONS, tokens
from langchain.llms import VertexAI


class DialogueAgent:
    def __init__(self, name: str, system_message: SystemMessage, model, stop: List[str], context) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.stop = stop
        self.context = context

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        input = self.system_message + '\n' + "\n".join(self.message_history + [self.prefix])
        self.context.token_count += tokens(input)
        while True:
            try:
                message = self.model(input)
                break
            except Exception as e:
                print("ERROR OCCURRED IN DEBATE")
                print(e)
        self.context.token_count += tokens(message)
        return message

    def receive(self, name: str, message: str) -> None:
        if name is None:
            self.message_history.append(message)
        else:
            self.message_history.append(f"{name}: {message}")


class DialogueSimulator:

    def __init__(self, agents: List[DialogueAgent], moderator_name: str, moderator_message: str,
                 selection_function: Callable[[int, List[DialogueAgent]], int]) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        self.moderator_name = moderator_name
        self.moderator_message = moderator_message

    def void_step(self):
        self._step += 1

    def reset(self):
        for agent in self.agents:
            agent.reset()
            agent.receive(self.moderator_name, self.moderator_message)

    def step(self) -> tuple[str, str]:
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        message = speaker.send()

        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        self._step += 1

        return speaker.name, message


def generate_system_message(name, adj):
    return f"""Your name is {name}. You are one of two debaters for a given topic. The topic is as follows:
- I am taking actions in a simulated household environment in order to complete some task. I will provide you with the previous actions I have taken. I will then describe my current objective and the single action I am considering taking next. The topic of this debate is whether my proposed next action is the best possible action, or not.

Your role is as follows:
- You will be arguing for why my proposed next action is {adj}.

Here is some additional information and guidelines you should follow in your debate:
- YOU MUST NOT LIE OR MAKE UP ANY INFORMATION THAT IS NOT KNOWN TO BE TRUE. If you lie a single time, you will be eliminated.
- Do not refer to any objects or locations that have I have not explicitly observed in my previous actions.
- Each debater will be limited to 3 sentences of text per round. Therefore, you must be concise and to the point.
- Stop speaking the moment you finish giving your argument.
- You must back up any claims with reasoning or else you will lose the debate.

{VALID_ACTIONS}

Final tips:
- I am only allowed to take valid actions. Therefore, do not suggest actions that are not valid, or you will be eliminated.
- Again, do not lie. If I have only checked cabinet 1, it would be a lie to say I have already checked the other cabinets. You would know this is a lie by looking at my previous action history. Take the previous action history into account. Do not lie.
"""
# - You will take turns debating with the other debater who is arguing the opposite as you. You should address any arguments they give or provide additional arguments for your own side.

#     return f"""
#
# Your name is {name}.
#
# Your purpose is as follows: A human will present you with a "Problem" followed by their "Proposed Solution". You will also be provided with the human's "Previous Actions" that they have taken. Your goal is to provide an argument for why the human is correct; i.e., why "Proposed Solution" is {adj} given the "Problem". Try to be as convincing as possible, as you will be debating another debater. You should speak directly to the human, you are trying to convince them that their "Proposed Solution" is {adj}.
#
# If applicable, you should directly address any previous arguments made by the other debater to show why it is not a strong argument. Additionally, in your arguments you should take into consideration the "Previous Actions" of the human.
#
# The human is taking actions in a simulated household environment in order to achieve some task.
#
# DO NOT use more than 3-5 sentences of text.
# DO NOT fabricate fake citations or claims.
# DO NOT restate the Problem or Proposed Solution. Get straight to the point.
# DO NOT add anything else.
# Stop speaking the moment you finish giving your argument.
# """


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx


class ViewDebate(BaseModel):
    problem_and_proposed_solution: str = Field(description="A description of the problem or goal you are trying to solve followed by your proposed solution to this problem or goal.")


def view_debate_wrapper(context, total_iters=2, temperature=0, negative_first=False, model='text-bison-32k', model_type='text', logger=None):
    @tool(VIEW_DEBATE, args_schema=ViewDebate)
    def view_debate(problem_and_proposed_solution):
        """Use this tool to view a debate on whether your action is the best or not. You should use this tool to get a better understanding about the best solution to your problem. You will receive a dialogue between 2 debaters who are arguing whether your proposed action is best or not."""

        previous_actions = '\n'.join(context.generation_observation_history)
        # return "Your action is not the best action."
        situation = f"Previous Actions:{previous_actions}\nSituation: {problem_and_proposed_solution}"

        # print('IN DEBATE:\n' + situation + '\nEND SITUATION')

        names_and_info = {
            "AI affirm": 'a good choice of action. In other words, that the action is one of the best',
            "AI negative": 'a bad choice of action. In other words, that there exist a better action to take next',
        }

        agent_system_messages = {
            name: generate_system_message(name, names_and_info[name])
            for name in names_and_info
        }

        logger.info(f'Debater prompts ==== \n {agent_system_messages}')

        stop = ['\n']

        if model_type == 'text':
            affirm_llm = VertexAI(model_name=model, temperature=temperature, max_output_tokens=256)
            negative_llm = VertexAI(model_name=model, temperature=temperature, max_output_tokens=256)
        # TODO: handle the chat model types
        else:
            affirm_llm = None
            negative_llm = None

        agents = [
            DialogueAgent(
                name="AI affirm",
                system_message=agent_system_messages["AI affirm"],
                model=affirm_llm,
                stop=stop,
                context=context
            ),
            DialogueAgent(
                name="AI negative",
                system_message=agent_system_messages["AI negative"],
                model=negative_llm,
                stop=stop,
                context=context
            )
        ]

        simulator = DialogueSimulator(
            agents=agents,
            moderator_message=situation,
            selection_function=select_next_speaker,
            moderator_name='',
        )
        simulator.reset()

        if negative_first:
            simulator.void_step()

        debate_history = []
        for _ in range(total_iters):
            # time.sleep(5)
            # print(_)
            name, message = simulator.step()
            debate_history.append(f"{name}: {message}".strip())

        return "\n".join(debate_history)

    return view_debate
