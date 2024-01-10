import sys
import os
import openai
from collections import deque
from typing import List
from dotenv import load_dotenv

# AEA dependencies
from aea.agent import Agent
from aea.configurations.base import SkillConfig
from aea.connections.base import Connection
from aea.identity.base import Identity
from aea.skills.base import Skill, SkillContext
from aea.skills.behaviours import FSMBehaviour, State
from aea.context.base import AgentContext

# import functions used to build the agent's actions
from actions import (
    task_creation_prompt_builder,
    task_creation_handler,
    task_prioritization_prompt_builder,
    task_prioritization_handler,
    task_execution_prompt_builder,
    task_execution_handler,
    task_stop_or_not_prompt_builder,
    task_stop_or_not_handler,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# flag to stop the procedure
STOP_PROCEDURE = False

# action types definition, each action type makes two function calls: builder & handler
# the initial action type is execution of the first task
initial = "task_execution_1"
action_types = {
    "task_creation": {
        "prompt_builder": task_creation_prompt_builder,
        "handler": task_creation_handler,
    },
    "task_prioritization": {
        "prompt_builder": task_prioritization_prompt_builder,
        "handler": task_prioritization_handler,
    },
    "task_execution_1": {
        "prompt_builder": task_execution_prompt_builder,
        "handler": task_execution_handler,
    },
    "task_execution_2": {
        "prompt_builder": task_execution_prompt_builder,
        "handler": task_execution_handler,
    },
    "task_stop_or_not": {
        "prompt_builder": task_stop_or_not_prompt_builder,
        "handler": task_stop_or_not_handler,
    },
}

# State Machine Definition
# Ending states, adds the option to stop the procedure and execute task_stop_or_not
if STOP_PROCEDURE:
    transitions = {
        "task_execution_1": {"done": "task_creation"},
        "task_creation": {"done": "task_execution_2"},
        "task_execution_2": {"done": "task_prioritization"},
        "task_prioritization": {"done": "task_stop_or_not"},
        "task_stop_or_not": {"done": "task_execution_1", "stop": None},
    }
# runtime state transitions of loop (execution, creation, execution, prioritization)
else:
    transitions = {
        "task_execution_1": {"done": "task_creation"},
        "task_creation": {"done": "task_execution_2"},
        "task_execution_2": {"done": "task_prioritization"},
        "task_prioritization": {"done": "task_execution_1"},
    }


class SimpleStateBehaviour(State):
    def act(self) -> None:
        """
        Act implementation.
        """
        # get the action type
        action_type = action_types[self.name]
        # get the prompt builder for the action type
        builder_ = action_type["prompt_builder"]
        # build the prompt using the shared state from the Agent's context
        prompt = builder_(self.context.shared_state)
        # use the prompt above to input into GPT to get the response
        response = self.openai_call(prompt)
        # get the handler for the action type
        handler_ = action_type["handler"]
        # get the event to trigger from the handler
        event_to_trigger = handler_(response, self.context.shared_state)
        self.executed = True
        self._event = event_to_trigger

    @staticmethod
    def openai_call(
        prompt: str,
        use_gpt4: bool = False,
        temperature: float = 0.5,
        max_tokens: int = 200,
    ):
        if use_gpt4:
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content.strip()
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].text.strip()

    def is_done(self) -> bool:
        """Get is done."""
        return self._event is not None


# instantiate FSMBehaviour class for use in constructing the agent's FSM transitions
class MyFSMBehaviour(FSMBehaviour):
    def setup(self):
        pass

    def teardown(self):
        pass


# create the agent's shared state and return it, "memory" is the shared state
# takes in string arguments to set first task and the objective of the agent
def create_memory(
    first_task: str,
    objective: str,
) -> dict:
    """Create the shared memory."""
    memory = {
        "objective": objective,
        "task_list": deque([]),
        "current_task": {},
        "result": {"data": first_task},
        "keep_going": True,
    }
    memory["task_list"].append({"id": 1, "name": first_task})
    return memory


def build_fsm_and_skill(memory: dict) -> tuple[MyFSMBehaviour, Skill]:
    """
    Build the FSM object and the Skill object. The FSM is built by loading
    all the Simple state behaviours and their respective transition
    functions into the FSM. The Skill object is built by updating the skill
    behaviours with the FSM behaviour after the states/transitions have
    been loaded into it.

    fsm, _ = build_fsm_and_skill(memory) ...to get the fsm object
    _, skill = build_fsm_and_skill(memory) ...to get the skill object

    Args:
        memory (dict): the agent's shared state

    Returns:
        tuple[MyFSMBehaviour, Skill]: the FSM object and the Skill object
    """
    # create skill and skill context
    config = SkillConfig(name="dummy", author="dummy")
    skill = Skill(configuration=config)
    skill_context = SkillContext(skill=skill)
    # create empty agent context to utilize shared state of Agent
    agent_context = AgentContext(
        identity=None,
        connection_status=None,
        outbox=None,
        decision_maker_message_queue=None,
        decision_maker_handler_context=None,
        task_manager=None,
        default_ledger_id=None,
        currency_denominations=None,
        default_connection=None,
        default_routing=None,
        search_service_address=None,
        decision_maker_address=None,
        data_dir=None,
    )
    # set the agent context
    skill_context.set_agent_context(agent_context)
    skill_context.shared_state.update(memory)
    # create the FSM object
    fsm = MyFSMBehaviour(name="babyAGI-loop", skill_context=skill_context)

    # load the states and transitions (SimpleStateBehaviour) into the FSM object
    for key in action_types.keys():
        if key not in transitions:
            continue
        behaviour = SimpleStateBehaviour(name=key, skill_context=skill_context)
        is_initial = key == initial
        fsm.register_state(str(behaviour.name), behaviour, initial=is_initial)
        for event, target_behaviour_name in transitions[key].items():
            fsm.register_transition(str(behaviour.name), target_behaviour_name, event)

    # update the skill behaviours with the FSM behaviour to build the skill
    skill.behaviours.update({fsm.name: fsm})

    return fsm, skill


class BabyAGI(Agent):
    """A re-implementation of the Baby AGI using the Open AEA framework."""

    def __init__(
        self,
        identity: Identity,
        memory: dict,
        connections: List[Connection] = None,
    ):
        """Initialise the agent."""
        super().__init__(identity, connections)
        fsm, _ = build_fsm_and_skill(memory)
        self.fsm = fsm

    def act(self):
        """Act implementation."""
        if self.fsm.is_done():
            print("done!")
            return
        self.fsm.act()

    def setup(self):
        # empty setup method
        pass

    def teardown(self):
        # empty teardown method
        pass


def run(first_task: str, objective: str):
    """
    Run babyAGI with the given first task + objective using
    open-aea's "Agent" & "FSMBehaviour" classes.

    Args:
        first_task (str): the first task to be completed by the agent
        objective (str): the objective of the agent
    """

    # Create the agent's shared state object
    memory = create_memory(first_task, objective)

    # Create an identity for the agent
    identity = Identity(
        name="baby_agi", address="my_address", public_key="my_public_key"
    )

    print("\033[89m\033[1m" + "\n===== Agent babyAGI ONLINE =====" + "\033[0m\033[0m")

    # Create our Agent (without connections)
    my_agent = BabyAGI(identity, memory)

    # Set the agent running in a different thread
    try:
        my_agent.start()
    except KeyboardInterrupt:
        # Shut down the agent
        my_agent.stop()


if __name__ == "__main__":
    _, first_task, objective = sys.argv
    try:
        run(first_task, objective)
    except KeyboardInterrupt:
        print("\033[89m\033[1m" + "\n======== EXIT ========" + "\033[0m\033[0m")
        pass
