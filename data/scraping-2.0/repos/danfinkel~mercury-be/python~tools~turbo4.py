import json
import os
import openai
import time
from typing import Callable, Dict, Any, List, Optional, Union, Tuple
import dotenv
from dataclasses import dataclass, asdict
from openai.types.beta import Thread, Assistant
from openai.types import FileObject
from openai.types.beta.threads.thread_message import ThreadMessage
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from datetime import date

from ..tools import llm
from ..tools.llmtypes import Chat, TurboTool

dotenv.load_dotenv()

class Turbo4:
    """
    Simple, chainable class for the OpenAI's GPT-4 Assistant APIs.
    """

    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI() # type: ignore
        self.map_function_tools: Dict[str, TurboTool] = {}
        self.current_thread_id = None
        self.thread_messages: List[ThreadMessage] = []
        self.local_messages = []
        self.published_messages: List[Chat] = []
        self.assistant_id = None
        self.polling_interval = (
            0.5  # Interval in seconds to poll the API for thread run completion
        )
        self.model = "gpt-4-1106-preview"

    @property
    def chat_messages(self) -> List[Chat]:
        return [
            Chat(
                from_name=msg.role,
                to_name="assistant" if msg.role == "user" else "user",
                message=str(llm.safe_get(msg.model_dump(), "content.0.text.value")), # type: ignore
                created=msg.created_at,
            )
            for msg in self.thread_messages
        ]

    @property
    def tool_config(self):
        return [tool.config for tool in self.map_function_tools.values()]

    # ------------- Additional Utility Functions -----------------

    def run_validation(self, validation_func: Callable):
        print(f"run_validation({validation_func.__name__})")
        validation_func()
        return self
    
    def spy_on_assistant(self, output_file: str):
        sorted_messages = sorted(
            self.chat_messages, key=lambda msg: msg.created, reverse=False
        )
        messages_as_json = [asdict(msg) for msg in sorted_messages]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(messages_as_json, f, indent=2)

        return self  
    
    def bank_results(self):
        import psycopg2 

        sorted_messages = sorted(
            self.chat_messages, key=lambda msg: msg.created, reverse=False
        )        

        # parse the roll up
        summary_message = sorted_messages[-1].message
        task = summary_message.split('PYTHON SCRIPT')[0].replace('Use these TABLE_DEFINITIONS to get data to support the analysis.\n\n','').replace('QUESTION: ','')
        python_code = summary_message.split('PYTHON SCRIPT:\n')[1].split('ANSWER')[0].replace("'","''")
        answer = summary_message.split('ANSWER:\n')[1]
        msg_cost, tokens = self._cost()
        ins_stmt = f"INSERT INTO workHistory.ai_tasks (task,pythoncode,result,est_cost,est_tokens) VALUES('{task}','{python_code}','{answer}','{msg_cost}','{tokens}')"

        pgPath = os.environ.get('DATABASE_URL')
        conn = psycopg2.connect(pgPath)
        cur = conn.cursor()       
        
        cur.execute(ins_stmt)

        conn.commit()
        cur.close()
        conn.close()

        return self

    def _cost(self):
        """
        Get the estimated cost and token usage for the current thread.

        https://openai.com/pricing

        Open questions - how to calculate retrieval and code interpreter costs?
        """

        retrival_costs = 0
        code_interpreter_costs = 0

        msgs = [
            llm.safe_get(msg.model_dump(), "content.0.text.value")
            for msg in self.thread_messages
        ]
        joined_msgs = " ".join(msgs) # type: ignore

        msg_cost, tokens = llm.estimate_price_and_tokens(joined_msgs)

        return msg_cost, tokens        

    
    def get_costs_and_tokens(self, output_file: str) -> Tuple[float, float]:
        """
        Get the estimated cost and token usage for the current thread.

        https://openai.com/pricing

        Open questions - how to calculate retrieval and code interpreter costs?
        """
        msg_cost, tokens = self._cost()

        with open(output_file, "w") as f:
            json.dump(
                {
                    "cost": msg_cost,
                    "tokens": tokens,
                },
                f,
                indent=2,
            )

        return self # type: ignore
    
    def get_conversation(self) -> list[Chat]:
        return sorted(
            self.chat_messages, key=lambda msg: msg.created, reverse=False
        )

    def print_conversion(self):
        print(sorted(
            self.chat_messages, key=lambda msg: msg.created, reverse=False
        ))
        return self
    
    def get_unpublished_msgs(self):
        unpublished = []
        for msg in self.chat_messages:
            if msg.message not in [x.message for x in self.published_messages]:
                unpublished.append(msg)
                self.published_messages.append(msg)
        return unpublished


    # ------------- CORE ASSISTANTS API FUNCTIONS -----------------

    def get_or_create_assistant(self, name: str, model: str = "gpt-4-1106-preview"):
        print(f"get_or_create_assistant({name}, {model})")
        # Retrieve the list of existing assistants
        assistants: List[Assistant] = self.client.beta.assistants.list().data

        # Check if an assistant with the given name already exists
        for assistant in assistants:
            if assistant.name == name:
                self.assistant_id = assistant.id

                # update model if different
                if assistant.model != model:
                    print(f"Updating assistant model from {assistant.model} to {model}")
                    self.client.beta.assistants.update(
                        assistant_id=self.assistant_id, model=model
                    )
                break
        else:  # If no assistant was found with the name, create a new one
            assistant = self.client.beta.assistants.create(model=model, name=name)
            self.assistant_id = assistant.id

        self.model = model

        create_msg = Chat(
                from_name="sys_admin",
                to_name="front_end",
                message=f"Good news your AI assistant {name} is here and ready to help!",
                created=date.today().strftime('%Y-%m-%d'), # type: ignore
        )

        self.published_messages.append(create_msg)
        return self, str({"user": create_msg.from_name, 
                          "assistant_name": name,
                          "assistant_id": self.assistant_id,
                          "content": create_msg.message}) + "\n"

    def set_instructions(self, instructions: str):
        print(f"set_instructions()")
        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created or retrieved. Call get_or_create_assistant() first."
            )
        # Update the assistant with the new instructions
        updated_assistant = self.client.beta.assistants.update(
            assistant_id=self.assistant_id, instructions=instructions
        )

        instruction_msg = Chat(
                from_name="sys_admin",
                to_name="front_end",
                message= f"The assistant has been instructed:\n\n {instructions}",
                created=date.today().strftime('%Y-%m-%d'), # type: ignore
        )
        self.published_messages.append(instruction_msg)

        return self, str({"user": instruction_msg.from_name, 
                          "assistent_id": self.assistant_id,
                          "content": instruction_msg.message}) + "\n"                

    def equip_tools(
        self, turbo_tools: List[TurboTool], equip_on_assistant: bool = False
    ):
        print(f"equip_tools({turbo_tools}, {equip_on_assistant})")
        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created or retrieved. Call get_or_create_assistant() first."
            )

        # Update the functions dictionary with the new tools
        self.map_function_tools = {tool.name: tool for tool in turbo_tools}

        if equip_on_assistant:
            # Update the assistant with the new list of tools, replacing any existing tools
            updated_assistant = self.client.beta.assistants.update(
                tools=self.tool_config, assistant_id=self.assistant_id # type: ignore
            )

        # updated_assistant = self.client.beta.assistants.update(
        #     tools=[{"type": "code_interpreter"}], assistant_id=self.assistant_id #type: ignore
        # )

        return self

    def make_thread(self):
        print(f"make_thread()")

        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created. Call create_assistant() first."
            )

        response = self.client.beta.threads.create()
        self.current_thread_id = response.id
        self.thread_messages = []
        return self, response.id

    def add_message(self, message: str, refresh_threads: bool = False):
        #print(f"add_message({message})")
        self.local_messages.append(message)
        self.client.beta.threads.messages.create(
            thread_id=self.current_thread_id, content=message, role="user" # type: ignore
        )
        if refresh_threads:
            self.load_threads()
        
        msg_to_ai = Chat(
                from_name="sys_admin",
                to_name="front_end",
                message=message,
                created=date.today().strftime('%Y-%m-%d'), # type: ignore
        )
        self.published_messages.append(msg_to_ai)

        return self, str({"user": msg_to_ai.from_name, 
                          "assistant_id": self.assistant_id,
                          "thread_id": self.current_thread_id,
                          "content": msg_to_ai.message}) + "\n"

    def load_threads(self):
        self.thread_messages = self.client.beta.threads.messages.list(
            thread_id=self.current_thread_id # type: ignore
        ).data

    def list_steps(self):
        print(f"list_steps()")
        steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.current_thread_id, # type: ignore
            run_id=self.run_id,
        )
        print("steps", steps)
        return steps

    def run_thread(self, toolbox: Optional[List[str]] = None):
        print(f"run_thread({toolbox})")
        if self.current_thread_id is None:
            raise ValueError("No thread has been created. Call make_thread() first.")
        if self.local_messages == []:
            raise ValueError("No messages have been added to the thread.")

        if toolbox is None:
            tools = None
        else:
            # get tools from toolbox
            tools = [self.map_function_tools[tool_name].config for tool_name in toolbox]

            # throw if tool not found
            if len(tools) != len(toolbox):
                raise ValueError(
                    f"Tool not found in toolbox. toolbox={toolbox}, tools={tools}. Make sure all tools are equipped on the assistant."
                )

        # refresh current thread
        self.load_threads()

        # Start the thread running
        run = self.client.beta.threads.runs.create(
            thread_id=self.current_thread_id,
            assistant_id=self.assistant_id, # type: ignore
            tools=tools, # type: ignore
        )
        self.run_id = run.id

        # Polling mechanism to wait for thread's run completion or required actions
        while True:
            # self.list_steps()

            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=self.current_thread_id, run_id=self.run_id
            )
            if run_status.status == "requires_action":
                tool_outputs: List[ToolOutput] = []
                if run_status.required_action:
                    for (
                        tool_call
                    ) in run_status.required_action.submit_tool_outputs.tool_calls:
                        tool_function = tool_call.function
                        tool_name = tool_function.name

                        # Check if tool_arguments is already a dictionary, if so, proceed directly
                        if isinstance(tool_function.arguments, dict):
                            tool_arguments = tool_function.arguments
                        else:
                            # Assume the arguments are JSON string and parse them
                            tool_arguments = json.loads(tool_function.arguments)

                        print(f"run_thread() Calling {tool_name}({tool_arguments})")

                        # Assuming arguments are passed as a dictionary
                        function_output = self.map_function_tools[tool_name].function(
                            **tool_arguments
                        )

                        tool_outputs.append(
                            ToolOutput(tool_call_id=tool_call.id, output=function_output)
                        )

                # Submit the tool outputs back to the API
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.current_thread_id,
                    run_id=self.run_id,
                    tool_outputs=[to for to in tool_outputs],
                )
            elif run_status.status == "completed":
                self.load_threads()
                new_msgs = self.get_unpublished_msgs()
                return self, new_msgs

            time.sleep(self.polling_interval)  # Wait a little before polling again

    def enable_retrieval(self):
        print(f"enable_retrieval()")
        if self.assistant_id is None:
            raise ValueError(
                "No assistant has been created or retrieved. Call get_or_create_assistant() first."
            )

        # Update the assistant with the new list of tools, replacing any existing tools
        updated_assistant = self.client.beta.assistants.update(
            tools=[{"type": "retrieval"}], assistant_id=self.assistant_id
        )

        return self