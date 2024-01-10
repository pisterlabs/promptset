import json
import os
import time
from typing import List

import openai
from openai.types.beta.threads import ThreadMessage
from PIL import Image

import agent.tools.github_tools as github_tools
import agent.tools.web_reader as web_reader
from agent.excecutor import FunctionExecutor
from agent.prompts import BASE_INSTRUCTION, STATUS_UPDATE
from agent.tools.github_tools import GitHubInterface


client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def build_frontend_developer_agent():
    tools = github_tools.get_tools()
    tools.extend(web_reader.get_tools())
    tools.append({"type": "code_interpreter"})
    assistant = client.beta.assistants.create(
        name="Serhii, the Frontend Developer",
        instructions=BASE_INSTRUCTION,
        tools=tools,
        model="gpt-4-1106-preview"
    )
    return assistant


def get_frontend_developer_agent():
    assistants = client.beta.assistants.list()
    for assistant in assistants:
        if assistant.name == "Serhii, the Frontend Developer":
            return assistant
    return build_frontend_developer_agent()


class FrontendAgentRunner:
    def __init__(self, verbose: bool = False):
        self.agent = get_frontend_developer_agent()
        github_interface = GitHubInterface.from_github_token(
            os.environ["GITHUB_TOKEN"], 
            repository=os.environ["GITHUB_REPOSITORY"]
        )
        web_reader_interface = web_reader.WebPageToolExecutor()
        self.executor = FunctionExecutor([github_interface, web_reader_interface], verbose=verbose)
        self.thread = client.beta.threads.create()
        self.verbose = verbose
    
    def run(self, text: str, image: Image = None) -> List[ThreadMessage]:
        # TODO: add image support
        if self.verbose:
            print(f"Running agent with input: {text}")
            print(f"Thread id: {self.thread.id}")
        
        message = client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=text
        )
        run = client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.agent.id,
            instructions=STATUS_UPDATE.template.format(
                status=self.executor.execute("getStatus")
            ),
        )
        while run.status != "completed":
            if run.status == "requires_action":
                if self.verbose:
                    print("Run requires action")
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    run_output = self.executor.execute(
                        tool_call.function.name, 
                        **json.loads(tool_call.function.arguments)
                    )
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": run_output if isinstance(run_output, str) else json.dumps(run_output)
                        }
                    )
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )  
                if self.verbose:
                    print("Submitted tool outputs")
            elif run.status == "failed":
                raise Exception(run.last_error.message) 
            else:
                time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
        messages = client.beta.threads.messages.list(
            thread_id=self.thread.id
        )
        if self.verbose:
            print(f"Agent finished with output: {messages}")
        return list(messages)
