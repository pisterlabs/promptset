"""
To be used for keeping track of the bot's state as a context.
"""

from datetime import datetime
from typing import List, Any
import os
from langchain.chat_models.openai import _convert_message_to_dict, _convert_dict_to_message
import json
from pydantic import BaseModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage


class AgentMemory():
    def __init__(self, k: int = 8):
        self.k = k
        self.memory = []

    def append(self, message: BaseMessage):
        self.memory.append(message)
        first_message = self.memory[0]

        # keep the last k messages in memory
        if len(self.memory) > self.k:
            self.memory = self.memory[-(self.k-1):]
            # insert first_message
            self.memory.insert(0, first_message)

    def __getitem__(self, index: int):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)


class BotState(BaseModel):

    id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    conv_hist: dict[str, AgentMemory] = {}
    current_agent_index: int = 0
    last_human_input: str = None
    products_list: List[Any] = []
    agent_names: List[str] = []

    # initialize from a username parameter
    def __init__(self, username: str = None, **data: Any):
        super().__init__(**data)
        if username is not None:
            try:
                self.load_from_file(f'conv_hist_{username}.json')
            except:
                self.id = username

    class Config:
        validate_assignment = True
        # To allow AgentMemory
        arbitrary_types_allowed = True

    def save_to_file(self, file_name: str = None):
        logs_dir = 'convs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        if file_name is None:
            file_name = f'conv_hist_{self.id}.json'
        file_path = os.path.join(logs_dir, file_name)

        data_dict = self.dict(by_alias=True)

        # Converting conv_hist messages to proper dict message format before saving
        temp_conv_dist = {}
        for agent_name, messages in self.conv_hist.items():
            temp_conv_dist[agent_name] = [
                _convert_message_to_dict(message) for message in messages]
        data_dict['conv_hist'] = temp_conv_dist

        with open(file_path, 'w', errors='ignore') as f:
            json_data = json.dumps(data_dict)
            f.write(json_data)

    def next_agent(self):
        self.current_agent_index += 1

    def load_from_file(self, file_name: str = None):
        file_path = os.path.join('convs', file_name)
        with open(file_path, 'r') as f:
            data: dict = json.loads(f.read())
            for key, value in data.items():
                if key == 'conv_hist':
                    # Converting conv_hist dict messages to BaseMessage format after loading
                    for agent_name, messages in value.items():
                        value[agent_name] = AgentMemory()
                        for message in messages:
                            value[agent_name].append(
                                _convert_dict_to_message(message))
                # Finally, set all attributes
                setattr(self, key, value)

    def get_conv_hist(self) -> List[dict]:
        conv_hist = []
        for agent_name in self.agent_names:
            for message in self.conv_hist[agent_name]:
                # check if message if of type AIMessage
                if isinstance(message, AIMessage) and message.additional_kwargs.get("function_call") is None:
                    conv_hist.append(_convert_message_to_dict(message))
                elif isinstance(message, HumanMessage):
                    conv_hist.append(_convert_message_to_dict(message))

        return conv_hist
