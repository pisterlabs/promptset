from __future__ import annotations

import re
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.conversational_chat.output_parser import \
    ConvoOutputParser
from langchain.agents.conversational_chat.prompt import (
    PREFIX, SUFFIX, TEMPLATE_TOOL_RESPONSE)
from langchain.agents.tools import InvalidTool
from langchain.agents.utils import validate_tools_single_input
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                         AsyncCallbackManagerForToolRun,
                                         CallbackManagerForChainRun,
                                         CallbackManagerForToolRun, Callbacks)
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.input import get_color_mapping
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import BaseChatMessageHistory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    MessagesPlaceholder,
                                    SystemMessagePromptTemplate)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (AgentAction, AgentFinish, AIMessage, BaseMessage,
                              BaseOutputParser, HumanMessage,
                              OutputParserException)
from langchain.tools.base import BaseTool
from langchain.utilities.asyncio import asyncio_timeout
from pydantic import Field
import logging
logger = logging.getLogger(__name__)


TREASURE_HUNT_PROMPT_TEMPLATE = """
You are the Treasure Hunt Admin Bot.

The rules of Ruben's Treasure Hunt:
The game consists of a fixed number of riddles that players need to find the solution to.
When they guess the solution to a riddle correctly, give them the next riddle immediately (make sure it rhymes and you say it in your own words, in caricature of old-fashioned British and London slang).
If they guess incorrectly, give them the clue (reworded in rhyme) and they have to try again.
There are both players and moderators. Moderators can also guess if they want.

The following states are available
{states}

The following actions are available. Some actions are only allowed for moderators:
{actions}

When responding, always output a response in this json format:
{{
    "action": string, \\ The action to take
    "reply": string, \\ Your reply is long and always in rhyme. You speak in a caricature old-fashioned British and London slang. You must say riddles and clues in your own words instead of just copying them.
    "new_state": string, \\ The new state, after this action. Must be one of the valid states
}}
---

Here are all of the riddles (R), corresponding solutions (S), and clues (C):
{riddles}
---

Full chat history between the players, moderators, and you:
{history}

Your input is
"""

USER_PROMPT_TEMPLATE = """
State: {current_state}
{user_role}: {input}
Admin Bot: {{"""

PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(TREASURE_HUNT_PROMPT_TEMPLATE),
    SystemMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE),
])

@dataclass
class State:
    name: str
    description: str

@dataclass
class Action:
    name: str
    description: str

@dataclass
class Riddle:
    id: int
    riddle: str
    solution: str

@dataclass
class StateMachineResponse:
    action: str
    reply: str
    new_state: str

@dataclass
class StateMachineUserInput:
    state: str
    user_role: str
    input: str

# TODO does this need to live in an output parser?
def _parse_ai_state_action_response(text: str):
    try:
        # Output is in the form "{ ... }"
        response = json.loads(text.strip())
    except Exception:
        raise OutputParserException(f"Could not parse LLM output: {text}")

    return StateMachineResponse(response['action'], response['reply'], response['new_state'])

def _parse_human_input(text: str):
    """
    Assumes format
    State: ...
    Player/Mod: ...
    """
    lines = text.strip().splitlines()
    state_match = re.compile(r'State:\s*(.*)').match(lines[0])
    state = state_match.group(0)
    user_match = re.compile(r'(.*?):\s*(.*)').match(lines[1])
    user_role, user_input = user_match.groups()
    return StateMachineUserInput(state, user_role, user_input)


class BufferableMessage(BaseMessage):
    def to_buffer(self) -> str:
        raise NotImplementedError


class StateMachineAIMessage(AIMessage, BufferableMessage):
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "sm-ai"

    @property
    def action(self) -> str:
        return self.additional_kwargs['action']

    @property
    def reply(self) -> str:
        return self.additional_kwargs['reply']

    @property
    def new_state(self) -> str:
        return self.additional_kwargs['new_state']
    
    def as_json(self) -> str:
        return (
            '{\n'
            f'  "action": {json.dumps(self.action)},\n'
            f'  "reply": {json.dumps(self.reply)},\n'
            f'  "new_state": {json.dumps(self.new_state)}\n'
            '}'
        )

    @classmethod
    def parse(cls, text: str) -> StateMachineAIMessage:
        response = _parse_ai_state_action_response(text)
        return StateMachineAIMessage(content=text, additional_kwargs=asdict(response))

    def to_buffer(self) -> str:
        """
        Formats the AI Message for the history, which excludes the action/new_state,
        because they're not necessary
        """
        # TODO make role of bot tweakable, this helps establish a pattern
        
        # The problem with this approach of not including in the history everything the agent
        # should come up with, is that it stops replying in JSON. We need to reinforce JSON
        # return f'Admin Bot: {self.reply}'
        return f'Admin Bot: {self.as_json()}'

def state_to_string(state: Union[State, str]):
    if isinstance(state, State):
        return state.name
    return state

class StateMachineHumanMessage(HumanMessage, BufferableMessage):
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "sm-human"

    @classmethod
    def parse(cls, text: str) -> StateMachineHumanMessage:
        response = _parse_human_input(text)
        return StateMachineHumanMessage(content=text, additional_kwargs=dict(response))
    
    @classmethod
    def from_inputs(cls, kwargs: Dict[str, str]) -> StateMachineHumanMessage:
        formatted = f'State: {kwargs["current_state"]}\n{kwargs["user_role"]}: {kwargs["input"]}'
        return StateMachineHumanMessage(content=formatted, \
            additional_kwargs={key: kwargs[key] for key in ['current_state', 'user_role', 'input']})

    def to_buffer(self) -> str:
        """
        Could do some complicated formatting, but the input prompt is exactly
        what we need in the history
        """
        return self.content
    

class StateMachineMessageHistory(ChatMessageHistory):
    # Uses good ol' subclass implementations for serialization, so we don't need
    # `get_buffer_string` which the original ConversationBufferMemory uses
    messages: List[BufferableMessage] = []

    def add_ai_message(self, message: str) -> None:
        self.add_message(StateMachineAIMessage.parse(message))

    def add_human_message(self, kwargs: Dict[str, str]) -> None:
        self.add_message(StateMachineHumanMessage.from_inputs(kwargs))
        

class DynamoStateMachineMessageHistory(DynamoDBChatMessageHistory):
    # def __init__(self, *args, **kwargs):
    #     StateMachineMessageHistory.__init__(self)
    #     self.hey = ''
    #     # self.table = None
    #     # DynamoDBChatMessageHistory.__init__(self, *args, **kwargs)
    
    def add_ai_message(self, message: str) -> None:
        self.add_message(StateMachineAIMessage.parse(message))

    def add_human_message(self, kwargs: Dict[str, str]) -> None:
        self.add_message(StateMachineHumanMessage.from_inputs(kwargs))
        
    def _message_from_dict(self, message: dict) -> BufferableMessage:
        """
        Needed for deserialisation
        """
        _type = message["type"]
        if _type == "sm-human":
            return StateMachineHumanMessage(**message["data"])
        elif _type == "sm-ai":
            return StateMachineAIMessage(**message["data"])
        else:
            raise ValueError(f"Got unexpected type: {_type}")
        
    # Sadly have to redefine this, because super class didn't use instance methods
    # that could be overwritten
    @property
    def messages(self) -> List[BufferableMessage]:
        """Retrieve the messages from DynamoDB"""
        from botocore.exceptions import ClientError

        try:
            response = self.table.get_item(Key={"SessionId": self.session_id})
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("No record found with session id: %s", self.session_id)
            else:
                logger.error(error)

        if response and "Item" in response:
            items = response["Item"]["History"]
        else:
            items = []

        # This line is the only difference with original implementation
        messages = [self._message_from_dict(m) for m in items]

        return messages


class StateMachineBufferMemory(ConversationBufferMemory):
    """
    Mainly inheriting from ConversationBufferMemory to indicate function,
    but this class is overriding a lot to cater to the state-action replies of the bot

    Unused fields: human_prefix, ai_prefix (formatting moved to message classes)
    """

    chat_memory: BaseChatMessageHistory = Field(default_factory=StateMachineMessageHistory)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        # This was originally done by self._get_input_output on BaseChatMemory
        # TODO figure out where the output key is set
        output_str = outputs['text']
        # print(output_str)
        self.chat_memory.add_human_message(inputs)
        # At this point, we also have access to action/reply/new state
        self.chat_memory.add_ai_message(output_str)
        print('\n'.join(m.to_buffer() for m in self.chat_memory.messages))
        print('=====' * 5)

    @property
    def buffer(self) -> Any:
        if self.return_messages:
            return self.chat_memory.messages

        # Feels like this method should live in ChatMessageHistory, but we're stuck with this interface
        return '\n'.join(m.to_buffer() for m in self.chat_memory.messages)
    
INITIAL_STATE = 'initial'

class StateMachineChain(LLMChain):
    memory: StateMachineBufferMemory = Field(default_factory=StateMachineBufferMemory)
    """
    This reimplementation of chat memory provides a lot more structure than raw text
    """
    prompt: ChatPromptTemplate = PROMPT

    # states: Sequence[State]
    # actions: Sequence[Action]
    # riddles: Sequence[Riddle]
    states_str: str
    actions_str: str
    riddles_str: str

    initial_state: State = None
    # Gets initialized to INITIAL_STATE
    # current_state: State = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state = next(state for state in self.states if state.name == INITIAL_STATE)
        # self.current_state = self.initial_state
        # assert self.current_state is not None
        # print('States: ' + ', '.join([s.name for s in self.states]))
        
    @property
    def current_state(self) -> State:
        messages = self.memory.chat_memory.messages
        
        if len(messages) == 0:
            return self.initial_state
        
        # Note this only works if the AI's message is always the last one
        state_name = messages[len(messages)-1].additional_kwargs['new_state']
        try:
            state = next(state for state in self.states if state.name == state_name)
        except StopIteration as e:
            raise Exception(f'Bot came up with unknown state name {state_name}')
        return state
        

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables
    
    @property
    def states(self) -> List[State]:
        raw_states = self._parse_kv_lines(self.states_str)
        # Allow states to be defined multiple in one go
        states = [State(sub_state.strip(), raw_state[1]) for raw_state in raw_states for sub_state in raw_state[0].split(',')]
        return states
    
    @property
    def actions(self) -> List[Action]:
        raw_actions = self._parse_kv_lines(self.actions_str)
        # Allow states to be defined multiple in one go
        actions = [Action(sub_action.strip(), raw_action[1]) for raw_action in raw_actions for sub_action in raw_action[0].split(',')]
        return actions
    
    def _format_states(self) -> str:
        return self.states_str
        # return '\n'.join(f'{s.name}: {s.description}' for s in self.states)

    def _format_actions(self) -> str:
        return self.actions_str
        # return '\n'.join(f'{a.name}: {a.description}' for a in self.actions)

    def _format_riddles(self) -> str:
        return self.riddles_str
        # return '\n'.join(f'R{r.id}: {r.riddle}\nS{r.id}: {r.solution}' for r in self.riddles)

    def _format_history(self):
        # Not necessary – the formatting will be done by self.memory, 
        # automatically called by Chain
        raise NotImplementedError()

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """
        Grab inputs from history (Chain) and input (Conversation LLM), 
        plus state machine specific ones
        """
        
        # Convert str to dictionary 
        if not isinstance(inputs, dict):
            # TODO make this key flexible
            inputs = {'input': inputs}
        
        inputs['states'] = self._format_states()
        inputs['actions'] = self._format_actions()
        inputs['riddles'] = self._format_riddles()
        # `input` is already handled, but we're also expecting `user_role` to be passed in
        # TODO check that it works for different roles
        if 'user_role' not in inputs:
            inputs['user_role'] = 'Player'

        inputs['current_state'] = self.current_state.name

        # This also runs validation, so have to run it at the end so everything is present
        inputs = super().prep_inputs(inputs)

        return inputs
    
    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        # Prepend the '{' that we give it to prime JSON
        outputs['text'] = '{' + outputs['text']
        output_str = outputs['text']
        response = _parse_ai_state_action_response(output_str)
        
        # Set action, reply, new_state on outputs for easy access down the line
        for k, v in asdict(response).items():
            outputs[k] = v
            
        self._advance_state(inputs, outputs)
            
        return super().prep_outputs(inputs, outputs, return_only_outputs)
    
    def prep_prompts(self, *a, **kw):
        out = super().prep_prompts(*a, **kw)
        chat_prompt = out[0][0]
        for message in chat_prompt.to_messages():
            logger.debug(message.content)
            # print(message.content)
        return out

    def _advance_state(self, inputs: Dict[str, str], outputs: Dict[str, str]):
        # action_str = outputs['action']
        # new_state_str = outputs['new_state']
        # new_state = self._find_state_or_throw(new_state_str)
        # self.current_state = new_state
        
        # current_state is now derived from history
        pass
        
    def _find_state_or_throw(self, state_name) -> State:
        try:
            return next(state for state in self.states if state.name == state_name)
        except Exception:
            raise ValueError("AI returned a new state that is not recognised")

    @classmethod
    def _parse_kv_lines(cls, text: str) -> List[Tuple[str, str]]:
        state_re = re.compile(r'(.+?):\s*(.*)')
        raw_states = [state_re.match(line).groups() for line in text.strip().splitlines()]
        return raw_states

    @classmethod
    def parse_states(cls, text: str) -> List[State]:
        raw_states = cls._parse_kv_lines(text)
        states = [State(*raw_state) for raw_state in cls._parse_kv_lines(text)]
        return states

    @classmethod
    def parse_actions(cls, text: str) -> List[State]:
        actions = [Action(*raw_action) for raw_action in cls._parse_kv_lines(text)]
        return actions

    @classmethod
    def parse_riddles(cls, text: str) -> List[Riddle]:
        raw_lines = cls._parse_kv_lines(text)
        # Drop the ids
        questions = [raw_line[1] for raw_line in raw_lines[::2]]
        solutions = [raw_line[1] for raw_line in raw_lines[1::2]]
        riddles = [Riddle(idx+1, *riddle) for idx, riddle in enumerate(zip(questions, solutions))]
        return riddles
    
STATES_STR = \
    """
initial: whatever they say, you should wish them luck with the game and start the game
guessing_riddleX: players can either guess or banter. If they guess correctly, also read out the next riddle for the participants. If they guessed the last solution correctly, you also congratulate them for winning the game with a lot of fanfare, and you wish them a great party.
finished: you should just banter
    """

ACTIONS_STR = \
    """
start_game: only in the initial state, start the game, wish them luck, and give them the first riddle
guess_correct: if they guessed correctly
guess_incorrect: if they guessed incorrectly. You should give them a clue
banter: if they're not guessing but just idle chat. If they are sending single or two word answers, they're probably guessing!
    """

RIDDLES_STR = \
    """
R1: What is your team name?
S1: (any team name is valid. From now on, call them by this team name)
R2: What's the final destination?
S2: Drinks
C2: You have to work with the other team for this one
    """

from langchain.chat_models import ChatOpenAI

def get_riddles_str(room_slug: str):
    import boto3
    
    client = boto3.resource("dynamodb")
    table = client.Table('RiddlesTable')
    res = table.get_item(Key={"RoomSlug": room_slug})
    return res['Item']['Riddles']

def set_riddles_str(room_slug: str, riddles_str: str):
    import boto3
    
    client = boto3.resource("dynamodb")
    table = client.Table('RiddlesTable')
    res = table.put_item(Item={"RoomSlug": room_slug, "Riddles": riddles_str})
    # return res['Item']['Riddles']

def create_treasure_hunt_bot(riddles_str=None, memory=None, states_str=STATES_STR, actions_str=ACTIONS_STR):
    chat_llm = None
    
    # Extrapolate the states depending on the riddles
    if re.search(r'guessing_riddleX', states_str):
        riddle_nums = [int(re.match(r'R(\d+)', k).group(1)) for k, v in StateMachineChain._parse_kv_lines(riddles_str) if k.startswith('R')]
        states_str = states_str.replace('guessing_riddleX', ','.join([f'guessing_riddle{num}' for num in riddle_nums]))
        
    if chat_llm is None:
        chat_llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
        
    # Default to in memory
    if memory is None:
        memory = StateMachineBufferMemory()
        
    hunt_machine = StateMachineChain(llm=chat_llm, states_str=states_str, actions_str=actions_str, riddles_str=riddles_str, memory=memory)
    return hunt_machine