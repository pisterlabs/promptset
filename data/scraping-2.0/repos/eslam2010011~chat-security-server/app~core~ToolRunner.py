import asyncio
import os
import re
import ssl
import subprocess
import uuid
from fastapi import WebSocket
from app.api.programs_api.BugcrowdManager import BugcrowdManager
from app.api.programs_api.HackerOneManager import HackerOneManager
from app.api.OpenaiManager import OpenaiManager
from app.core.JobManager import JobManager
from app.core.SessionManager import SessionManager
from app.core.SubprocessManager import SubprocessManager
from app.core.UpdateManager import UpdateManager
from app.core.YAMLFileManager import YAMLFileManager

file_path_programs = os.path.join('resources', 'programs.yaml')


class ToolRunner:
    def __init__(self, yaml_file, ROOT_DIR, ROOT_Folders):
        self.yaml_file = yaml_file
        self.ROOT_DIR = ROOT_DIR
        self.ROOT_Folders = ROOT_Folders
        self.session_manager = SessionManager()
        self.manager = SubprocessManager()
        self.file_manager = YAMLFileManager(self.yaml_file)
        self.programs = YAMLFileManager(file_path_programs)
        self.openai_manager = OpenaiManager("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")
        self.session_id = ""
        self.jobmanager = JobManager()
        self.update_manager = UpdateManager(self.file_manager)
        self.ssl_context = ssl.SSLContext()

    def sessionId(self, session_id):
        self.session_id = session_id

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['ssl_context']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ssl_context = ssl.SSLContext()

    def get_session_manager(self):
        return self.session_manager

    def get_programs(self):
        return self.programs

    async def install_tool(self, tool_name, websocket: WebSocket):
        for data in self.file_manager.search_name_tools(tool_name):
            shell = data["install"]
            if isinstance(shell, list):
                for s in shell:
                    await self.read_subprocess_output(s.format(path=self.ROOT_DIR), websocket)
            elif isinstance(shell, str):
                await self.read_subprocess_output(shell.format(path=self.ROOT_DIR), websocket)

    async def run_tool(self, tool_name, websocket: WebSocket):
        for data in self.file_manager.search_name_tools(tool_name):
            domain = await websocket.receive_text()
            run = data["command"].format(domain=domain, path=self.ROOT_DIR)
            await self.read_subprocess_output(run.format(domain=domain, path=self.ROOT_DIR), websocket)

    def get_Tools(self):
        return self.file_manager

    async def read_subprocess_output(self, command, websocket: WebSocket):
        chatID = self.session_manager.createChat(sessionId_=self.session_id, question=command)
        # await websocket.close(1000,"fdsfdsf")
        match_path = re.search(r'{path}', command)
        if match_path:
            command = command.format(path=self.ROOT_DIR)
        match_path_script = re.search(r'{script}', command)
        if match_path_script:
            command = command.format(script=self.ROOT_Folders)

        moderation_id = str(uuid.uuid4())
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True
        )

        match_path_input = re.search(r'i:', command)
        if match_path_input:
            print(command)
            process.stdin.write(command.encode())
            process.stdin.write_eof()

        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_without_color = re.sub(r'\x1b\[\d+m', '', line.decode('utf-8').strip())
            await websocket.send_json({"moderation_id": moderation_id, "data": line_without_color})
            self.session_manager.updateChat(self.session_id, chatID, line_without_color)
            # user_input = await asyncio.wait_for(websocket.receive_json(), timeout=1)
            # print(user_input)
            # process.stdin.write(user_input["command"].encode() + b"\n")
            # await process.stdin.drain()
            # #
            # process.stdin.write(user_input.encode() + b"\n")
            # await process.stdin.drain()

        # process.stdin.write_eof()
        await process.wait()

    async def read_subprocess_output_without_websocket(self, session_id, command):
        chatID = self.session_manager.createChat(sessionId_=session_id, question=command)
        match_path = re.search(r'{path}', command)
        if match_path:
            command = command.format(path=self.ROOT_DIR)
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True
        )
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_without_color = re.sub(r'\x1b\[\d+m', '', line.decode('utf-8').strip())
            self.session_manager.updateChat(session_id, chatID, line_without_color)
        await process.wait()

    async def openAiSendMessage(self, command, websocket: WebSocket):
        chatID = self.session_manager.createChat(sessionId_=self.session_id, question=command)
        moderation_id = str(uuid.uuid4())
        line = self.openai_manager.sendMessage(command)
        self.session_manager.updateChat(self.session_id, chatID, line)
        await websocket.send_json({"moderation_id": moderation_id, "data": line})

    async def run(self, command, websocket: WebSocket):
        text = command["command"]
        index_platform = text.find("Search Platform:")
        index_openai = text.find("Ask OpenAi:")
        # match_path = re.search(r'{path}', text)
        # if match_path:
        #     path = self.ROOT_Folders + f"/{self.session_id}"
        #     text = text.format(path=path)
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        if index_openai != -1:
            extracted_text = text[index_openai + len("Ask OpenAi:"):].strip()
            await self.openAiSendMessage(extracted_text, websocket)
        elif index_platform != -1:
            extracted_text = text[index_platform + len("Search Platform:"):].strip()
            search_program = self.search_program(extracted_text)
            if len(search_program) > 0:
                search_program_one = search_program[0]
                if search_program_one["platform"] == "bugcrowd":
                    await BugcrowdManager(self.session_manager, self.session_id).getScope(extracted_text, websocket)
                if search_program_one["platform"] == "hackerone":
                    await HackerOneManager(self.session_manager, self.session_id).getScope("/" + extracted_text,
                                                                                           websocket)
        else:
            await self.read_subprocess_output(text, websocket)

    def search_program(self, name):
        return self.programs.search_key_value("name", name)

    def add_job(self, hour, minute, command, jobName):
        return self.jobmanager.add_job(fun=self.read_subprocess_output_without_websocket, hour=hour, minute=minute,
                                       jobName=jobName,
                                       param=[self.session_id, command])

    def remove_job(self, job_id):
        self.jobmanager.remove_job(job_id)

    def update_tools(self):
        self.update_manager.update()

    def removeSession(self):
        self.session_manager.removeSession(sessionId_=self.session_id)
