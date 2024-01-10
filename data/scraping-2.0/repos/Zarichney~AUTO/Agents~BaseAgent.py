# /Agents/Agent.py

import textwrap
import time
from openai.types.beta.assistant import Assistant
from Utilities.Log import Debug, Log, type
from Utilities.Config import current_model
from typing import TYPE_CHECKING
from Agency.Arsenal import SHARED_TOOLS

if TYPE_CHECKING:
    from Agency.Agency import Agency

class BaseAgent:
    def __init__(self, agency:'Agency', assistant_id=None):
        if not agency:
            raise Exception("Agency not supplied")
        
        self.agency = agency
            
        self.tool_definitions = []
        if not hasattr(self, 'custom_tools'):
            self.custom_tools = []
        for tool in self.custom_tools:
            self.tool_definitions.append({"type": "function", "function": tool.openai_schema})
        for tool in SHARED_TOOLS:
            self.tool_definitions.append({"type": "function", "function": tool.openai_schema})
            
        if not hasattr(self, 'custom_instructions'):
            Log(type.Error, f"Agent {self.name} does not have a custom_instructions attribute")
            self.custom_instructions = ""
            
        # Create agent if it doesn't exist
        if assistant_id is None:
            
            # Standard template for all agents
            self.instructions = textwrap.dedent(f"""
            # Name
            {self.name}

            ## Description
            {self.description}
            
            {self.custom_instructions}

            ## Services You Offer:
            {self.services}
            
            {self.agency.get_team_instruction()}
            """).strip()
            
            assistant = self.agency.client.beta.assistants.create(
                name = self.name,
                description = self.description,
                instructions = self.instructions,
                model = current_model,
                metadata = {"services": self.services},
                tools = self.tool_definitions
            )
            
        else:
            assistant = self.agency.client.beta.assistants.retrieve(assistant_id=assistant_id)
        
        self.assistant:Assistant = assistant
        self.id = self.assistant.id
        self.instructions = self.assistant.instructions

        self.waiting_on_response = False
        self.task_delegated = False

        self.toolkit = []
        self._setup_tools()
    
    def add_tool(self, tool):
        self.toolkit.append(tool)

    def _setup_tools(self):
        # Add custom tools
        for tool in self.custom_tools:
            self.add_tool(tool)
            
        # Add communal tools
        for tool in SHARED_TOOLS:
            self.add_tool(tool)

    def get_completion(self, message=None, useTools=True):

        client = self.agency.client
        thread = self.agency.thread

        if self.agency.running_tool:
            Debug(f"Agent called for completion while it's currently waiting on tool usage to complete. Falling back to thread-less completion")
            return client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": message}
                ]
            ).choices[0].message.content
        else:
            # Messages can't be added while a tool is running so they get queued up
            # Unload message queue
            for queued_message in self.agency.message_queue:
                self.agency.add_message(message=queued_message)
            self.agency.message_queue = []

        self.waiting_on_response = False

        if message is not None:
            message = self.agency.add_message(message=message)

        # run creation
        if useTools:
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.id,
            )
        else:
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.id,
                tools=[] # forces assistant to respond with prompt
            )

        while True:
            # wait until run completes
            while run.status in ["queued", "in_progress"]:
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                time.sleep(1)

            # Assistant requested to have a tool executed
            if run.status == "requires_action":
                
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                if len(tool_calls) > 1:
                    Debug(f"{self.name} is invoking {len(tool_calls)} tool calls")
                
                for tool_call in tool_calls:
                    
                    tool_name = tool_call.function.name
                    Debug(f"{self.name} is invoking tool: {tool_name}")
                    
                    # Find the tool to be executed
                    tool_function = next((func for func in self.toolkit if func.__name__ == tool_name), None)
                    
                    if tool_function is None:
                        tool_names = [func.__name__ for func in self.toolkit]
                        Log(type.ERROR, f"No tool found with name {tool_name}. Available tools: {', '.join(tool_names)}")
                        output = f"{tool_name} is not a valid tool name. Available tools: {', '.join(tool_names)}"
                        
                    else:
                        self.agency.running_tool = True
                        try:
                            arguments = tool_call.function.arguments.replace('true', 'True').replace('false', 'False')
                            
                            output = tool_function(**eval(arguments)).run(agency=self.agency)
                            
                            Debug(f"Tool '{tool_name}' Completed.")

                        except Exception as e:
                            Log(type.ERROR, f"Error occurred in function '{tool_name}': {str(e)}")
                            output = f"Tool '{tool_name}' failed. Error: {str(e)}"
                        self.agency.running_tool = False

                    tool_outputs.append({"tool_call_id": tool_call.id, "output": output})

                if len(tool_calls) == 1:
                    Debug(f"Submitting tool output")
                else:
                    Debug(f"Submitting {len(tool_calls)} tool outputs")
                    
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                )

            # error
            elif run.status == "failed":
                Log(type.ERROR, f"Run Failed. Error: {run.last_error}")
                Debug(f"Run {run.id} has been cancelled")
                return "An internal server error occurred. Try again"

            # return assistant response
            else:
                completion = client.beta.threads.messages.list(thread_id=thread.id)
                
                # todos:
                # 1. send to agency
                # 2. store to disk
                # 3. use for new caching mechanism
                # 3.1 that can work across different threads (dictionary of hashes?)
                # 3.2 only store the prompt after 
                
                response = completion.data[0].content[0].text.value

                if self.task_delegated:
                    self.waiting_on_response = False
                else:
                    self.waiting_on_response = True

                return response
