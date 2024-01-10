import openai
import langchain
import json
import websockets
import asyncio
import os
import time
from nbformat.v4 import new_notebook
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool, StructuredTool, ShellTool
from langchain.schema.messages import SystemMessage, AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains import LLMChain
from typing import List, Dict, Any, Union, Optional
from langchain.pydantic_v1 import BaseModel, Field
from notebook.services.contents.filemanager import FileContentsManager
from langchain.agents import AgentExecutor

from .prompt import *
from .callback import DefaultCallbackHandler, PrintCallbackHandler
from .agent import OpenAIMultiFunctionsAgent

import traceback
import tracemalloc
tracemalloc.start()

langchain.debug = True


class SharedState:
    def __init__(self):
        self.answer = None
        self.has_answer = asyncio.Event()

class CreateNewNotebookInput(BaseModel):
    filename: str = Field(description="Required filename of the notebook.")

class ReadCellInput(BaseModel):
    index: int = Field(description="Required index of the cell to be read.")
    filename: Optional[str] = Field(description="Optional filename of the notebook to read from. If no filename is given, the active notebook will be used.")

class InsertCodeCellInput(BaseModel):
    index: int = Field(description="Required index of where to insert the cell.")
    code: str = Field(description="Required code to be inserted.")
    filename: Optional[str] = Field(description="Optional filename of the notebook to insert code into. If no filename is given, the active notebook will be used.")

class InsertMarkdownCellInput(BaseModel):
    index: int = Field(description="Required index of where to insert the cell.")
    text: str = Field(description="Required markdown text to be inserted.")
    filename: Optional[str] = Field(description="Optional filename of the notebook to insert markdown into. If no filename is given, the active notebook will be used.")

class EditCodeCellInput(BaseModel):
    index: int = Field(description="Required index of which cell to edit.")
    code: str = Field(description="Required code to be inserted.")
    filename: Optional[str] = Field(description="Optional filename of the notebook to edit. If no filename is given, the active notebook will be used.")

class EditMarkdownCellInput(BaseModel):
    index: int = Field(description="Required index of which cell to edit.")
    text: str = Field(description="Required markdown text to be inserted.")
    filename: Optional[str] = Field(description="Optional filename of the notebook to edit. If no filename is given, the active notebook will be used.")

class RunCellInput(BaseModel):
    index: int = Field(description="Required index of which cell to run.")
    filename: Optional[str] = Field(description="Optional filename of the notebook to run cell in. If no filename is given, the active notebook will be used.")

class DeleteCellInput(BaseModel):
    index: int = Field(description="Required index of which cell to delete.")
    filename: Optional[str] = Field(description="Optional filename of the notebook to delete cell from. If no filename is given, the active notebook will be used.")

class ReadNotebookInput(BaseModel):
    filename: Optional[str] = Field(description="Optional filename of the notebook to read. If no filename is given, the active notebook will be used.")


class MyContentsManager(FileContentsManager):
    def __init__(self, **kwargs):
        super(MyContentsManager, self).__init__(**kwargs)

    def create_notebook(self, path):
        # Create an empty notebook
        nb = new_notebook()
        nb.metadata.kernelspec = {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        }
        super().save({"type": "notebook", "content": nb}, path)

class Terminal(object):

    def __init__(self):
        self.agent = None
        self.chat_history_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        self.create_notebook_state = SharedState()
        self.read_cell_state = SharedState()
        self.insert_code_state = SharedState()
        self.insert_markdown_state = SharedState()
        self.edit_code_cell_state = SharedState()
        self.edit_markdown_cell_state = SharedState()
        self.run_code_cell_state = SharedState()
        self.delete_cell_state = SharedState()
        self.read_notebook_state = SharedState()

    async def start(self):
        print("starting terminal backend")
        self.primary_ws = await websockets.serve(self.primary_web_socket, "0.0.0.0", 8080)
        self.secondary_ws = await websockets.serve(self.secondary_web_socket, "0.0.0.0", 8081)
        await self.primary_ws.wait_closed()

    def create_agent(self, websocket, model, temp, openai_api_key):
        model += "-0613" # Better functions calling model
        self.model = model
        self.temp = temp

        callback=DefaultCallbackHandler(websocket)
        llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=temp, streaming=True, callbacks=[callback])
        tools = [
            ShellTool(name="shell_tool"),
            self.get_create_new_notebook_tool(websocket),
            self.get_read_cell_tool(websocket),
            self.get_insert_code_cell_tool(websocket),
            self.get_insert_markdown_cell_tool(websocket),
            self.get_edit_code_cell_tool(websocket),
            self.get_edit_markdown_cell_tool(websocket),
            self.get_run_code_cell_tool(websocket),
            self.get_delete_cell_tool(websocket),
            self.get_read_notebook_summary_tool(websocket)
        ]

        extra_prompt_messages = [
            SystemMessage(content=f"The current time and date is {time.strftime('%c')}"),
            MessagesPlaceholder(variable_name="memory"),
            SystemMessage(content="Let's work the following out in a step by step way to be sure we have the right answer. Let's first understand the problem and devise a plan to solve the problem.")
        ]

        prompt = OpenAIMultiFunctionsAgent.create_prompt(system_message=SystemMessage(content=agent_system_message), extra_prompt_messages=extra_prompt_messages)
        agent = OpenAIMultiFunctionsAgent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            max_iterations=15, 
            verbose=True,
            handle_parsing_errors=True
        )
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            return_intermediate_steps=False,
            handle_parsing_errors=True,
            memory=self.chat_history_memory
        )

    async def primary_web_socket(self, websocket, path):
        async for message in websocket:
            print("primary_web_socket received message", message)
            data = json.loads(message)
            await self.handle_primary_ws_response(websocket, data)

    async def handle_primary_ws_response(self, websocket, data):
        if data.get("method") == "clear":
            self.chat_history_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
        else:
            self.create_agent(websocket, data["model"], data["temp"], data["openai_api_key"])
            try:
                await self.agent.arun(data["message"])
            except Exception as e:
                msg = "Server error encountered during execution: " + str(e)
                self.chat_history_memory.save_context({"input": data["message"]}, {"output": msg})
                traceback.print_exc()
                response = {
                    "method": "systemError",
                    "message": msg
                }
                await websocket.send(json.dumps(response))
                data["message"] = """An error occured while running the previously tool. Try again, 
but make sure to conform to the function calling format and validate the input to the tool."""
                await self.handle_primary_ws_response(websocket, data)

    async def secondary_web_socket(self, websocket, path):
        async for message in websocket:
            print("secondary_web_socket received message", message)
            data = json.loads(message)
            if path == "/openNotebook":
                self.create_notebook_state.answer = data
                self.create_notebook_state.has_answer.set()
            elif path == "/readCell":
                self.read_cell_state.answer = data
                self.read_cell_state.has_answer.set()
            elif path == "/insertCodeCell":
                self.insert_code_state.answer = data
                self.insert_code_state.has_answer.set()
            elif path == "/insertMarkdownCell":
                self.insert_markdown_state.answer = data
                self.insert_markdown_state.has_answer.set()
            elif path == "/editCodeCell":
                self.edit_code_cell_state.answer = data
                self.edit_code_cell_state.has_answer.set()
            elif path == "/editMarkdownCell":
                self.edit_markdown_cell_state.answer = data
                self.edit_markdown_cell_state.has_answer.set()
            elif path == "/runCode":
                self.run_code_cell_state.answer = data
                self.run_code_cell_state.has_answer.set()
            elif path == "/deleteCell":
                self.delete_cell_state.answer = data
                self.delete_cell_state.has_answer.set()
            elif path == "/readNotebook":
                self.read_notebook_state.answer = data
                self.read_notebook_state.has_answer.set()
            else:
                print(f"secondary_web_socket() - path {path} not recognized")

    def get_create_new_notebook_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda filename: self.create_new_notebook_tool(default_ws, filename),
            coroutine=lambda filename: self.create_new_notebook_tool(default_ws, filename),
            name="create_new_notebook_tool",
            description="""Useful when you want to start a new project with a Jupyter notebook and set it as the active notebook.
You should enter the filename (remember to use ".ipynb" extension) of the notebook.""",
            args_schema=CreateNewNotebookInput
        )

    async def create_new_notebook_tool(self, default_ws, filename):
        try:
            mgr = MyContentsManager()
            mgr.create_notebook(filename)

            request = {
                "request": {"filename": filename}, 
                "start": True, 
                "method": "openNotebook"
            }
            await default_ws.send(json.dumps(request))
            await self.create_notebook_state.has_answer.wait()
            answer = self.create_notebook_state.answer
            self.create_notebook_state = SharedState()
            return answer["message"]
        except Exception as e:
            traceback.print_exc()
            return f"Notebook with filename: {filename} failed to be created. Ask user for what to do next."

    def get_read_cell_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda index, filename=None: self.read_cell_tool(default_ws, index, filename),
            coroutine=lambda index, filename=None: self.read_cell_tool(default_ws, index, filename),
            name="read_cell_tool",
            description="""Useful when you want to read the conent of a cell of a Jupyter notebook. If no filename is given the active notebook will be used.
This tool cannot read files, only cells from a jupyter notebook.
You should enter the index of the cell you want to read.""",
            args_schema=ReadCellInput
        )

    async def read_cell_tool(self, default_ws, index, filename):
        try:
            request = {
                "request": {"index": index, "filename": filename}, 
                "start": True, 
                "method": "readCell"
            }
            await default_ws.send(json.dumps(request))
            await self.read_cell_state.has_answer.wait()
            answer = self.read_cell_state.answer
            self.read_cell_state = SharedState()
            return answer["message"]
        except Exception as e:
            return "ERROR: " + str(e)

    def get_insert_code_cell_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda code, index, filename=None: self.insert_code_cell_tool(default_ws, code, index, filename),
            coroutine=lambda code, index, filename=None: self.insert_code_cell_tool(default_ws, code, index, filename),
            name="insert_code_cell_tool",
            description="""Useful when you want to insert a code cell in a Jupyter notebook. If no filename is given the active notebook will be used.
You should enter code and index of the cell you want to insert.""",
            args_schema=InsertCodeCellInput
        )

    async def insert_code_cell_tool(self, default_ws, code, index, filename):
        try:
            request = {
                "request": {"index": index, "code": code, "filename": filename}, 
                "start": True, 
                "method": "insertCodeCell"
            }
            await default_ws.send(json.dumps(request))
            await self.insert_code_state.has_answer.wait()
            answer = self.insert_code_state.answer
            self.insert_code_state = SharedState()
            return answer["message"]
        except Exception as e:
            return "ERROR: " + str(e)

    def get_insert_markdown_cell_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda text, index, filename=None: self.insert_markdown_cell_tool(default_ws, text, index, filename),
            coroutine=lambda text, index, filename=None: self.insert_markdown_cell_tool(default_ws, text, index, filename),
            name="insert_markdown_cell_tool",
            description="""Useful when you want to insert a mardkown cell in a Jupyter notebook. If no filename is given the active notebook will be used.
You should enter markdown text and index of the cell you want to insert.""",
            args_schema=InsertMarkdownCellInput
        )
    
    async def insert_markdown_cell_tool(self, default_ws, text, index, filename):
        try:
            request = {
                "request": {"index": index, "text": text, "filename": filename}, 
                "start": True, 
                "method": "insertMarkdownCell"
            }
            await default_ws.send(json.dumps(request))
            await self.insert_markdown_state.has_answer.wait()
            answer = self.insert_markdown_state.answer
            self.insert_markdown_state = SharedState()
            return answer["message"]
        except Exception as e:
            return "ERROR: " + str(e)

    def get_edit_code_cell_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda code, index, filename=None: self.edit_code_cell_tool(default_ws, code, index, filename),
            coroutine=lambda code, index, filename=None: self.edit_code_cell_tool(default_ws, code, index, filename),
            name="edit_code_cell_tool",
            description="""Useful when you want to edit a code cell in a Jupyter notebook. If no filename is given the active notebook will be used.
You must always enter the code and the index of the cell to be edited.
You should enter the code to be inserted and the index of the cell you want to edit.""",
            args_schema=EditCodeCellInput
        )

    async def edit_code_cell_tool(self, default_ws, code, index, filename):
        try:
            request = {
                "request": {"index": index, "code": code, "filename": filename}, 
                "start": True, 
                "method": "editCodeCell"
            }
            await default_ws.send(json.dumps(request))
            await self.edit_markdown_cell_state.has_answer.wait()
            answer = self.edit_markdown_cell_state.answer
            self.edit_markdown_cell_state = SharedState()
            return answer["message"]
        except Exception as e:
            return "ERROR: " + str(e)
        
    def get_edit_markdown_cell_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda text, index, filename=None: self.edit_markdown_cell_tool(default_ws, text, index, filename),
            coroutine=lambda text, index, filename=None: self.edit_markdown_cell_tool(default_ws, text, index, filename),
            name="edit_markdown_cell_tool",
            description="""Useful when you want to edit a markdown cell in a Jupyter notebook. If no filename is given the active notebook will be used.
You must always enter the markdown text and the index of the cell to be edited.
You should enter the markdown text to be inserted and the index of the cell you want to edit.""",
            args_schema=EditMarkdownCellInput
        )

    async def edit_markdown_cell_tool(self, default_ws, text, index, filename):
        try:
            request = {
                "request": {"index": index, "text": text, "filename": filename}, 
                "start": True, 
                "method": "editMarkdownCell"
            }
            await default_ws.send(json.dumps(request))
            await self.edit_code_cell_state.has_answer.wait()
            answer = self.edit_code_cell_state.answer
            self.edit_code_cell_state = SharedState()
            return answer["message"]
        except Exception as e:
            return "ERROR: " + str(e)

    def get_run_code_cell_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda index, filename=None: self.run_code_cell_tool(default_ws, index, filename),
            coroutine=lambda index, filename=None: self.run_code_cell_tool(default_ws, index, filename),
            name="run_code_cell_tool",
            description="""Useful when you want to run a code cell in a Jupyter notebook. If no filename is given the active notebook will be used.
The tool outputs the result of the execution. You should enter the index of the cell you want to run.""",
            args_schema=RunCellInput
        )

    async def run_code_cell_tool(self, default_ws, index, filename):
        try:
            request = {
                "request": {"index": index, "filename": filename}, 
                "start": True,
                "method": "runCode",
            }
            await default_ws.send(json.dumps(request))
            await self.run_code_cell_state.has_answer.wait()
            answer = self.run_code_cell_state.answer
            self.run_code_cell_state = SharedState()
            return answer["message"]
        except Exception as e:
            traceback.print_exc()
            return "ERROR: " + str(e)

    def get_delete_cell_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda index, filename=None: self.delete_cell_tool(default_ws, index, filename),
            coroutine=lambda index, filename=None: self.delete_cell_tool(default_ws, index, filename),
            name="delete_cell_tool",
            description="""Useful when you want to delete a code cell in a Jupyter notebook. If no filename is given the active notebook will be used.
This tool cannot delete files! You should enter the index of the cell you want to delete.""",
            args_schema=DeleteCellInput
        )

    async def delete_cell_tool(self, default_ws, index, filename):
        try:
            request = {
                "request": {"index": index, "filename": filename}, 
                "start": True, 
                "method": "deleteCell"
            }
            await default_ws.send(json.dumps(request))
            await self.delete_cell_state.has_answer.wait()
            answer = self.delete_cell_state.answer
            self.delete_cell_state = SharedState()
            return answer["message"]
        except Exception as e:
            traceback.print_exc()
            return "ERROR: " + str(e)

    def get_read_notebook_summary_tool(self, default_ws):
        return StructuredTool.from_function(
            func=lambda filename=None: self.read_notebook_summary_tool(default_ws, filename),
            coroutine=lambda filename=None: self.read_notebook_summary_tool(default_ws, filename),
            name="read_notebook_summary_tool",
            description="""Useful when you want to get a summary of the whole notebook to see whats in each cell and its outputs.
If you give no filename the active notebook will be used.You should enter the filename of the notebook.""",
            args_schema=ReadNotebookInput
        )

    async def read_notebook_summary_tool(self, default_ws, filename):
        try:
            request = {
                "request": {"filename": filename}, 
                "start": True, 
                "method": "readNotebook"
            }
            await default_ws.send(json.dumps(request))
            await self.read_notebook_state.has_answer.wait()
            answer = self.read_notebook_state.answer
            self.read_notebook_state = SharedState()
            
            llm = ChatOpenAI(model=self.model, temperature=self.temp)
            prompt_template = PromptTemplate(input_variables=["notebook"], template=read_notebook_summary_template)
            chain = LLMChain(
                llm=llm,
                prompt=prompt_template,
                verbose=True
            )
            return chain({"notebook": answer["message"]})
        except Exception as e:
            traceback.print_exc()
            return "ERROR: " + str(e)

