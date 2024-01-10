

import asyncio
import subprocess
from typing import Optional 
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain.tools import BaseTool 
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class FfmpegTool(BaseTool):
    name = "FfmpegTool"
    description = "useful for when you need to edit and manipulate audio files with ffmpeg"

    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        template = """Question: {query}

        Answer: what's a valid ffmpeg command to do this?"""

        prompt = PromptTemplate(template=template, input_variables=["query"])
        llm = OpenAI()
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        llm_output: str = llm_chain.run(query)

        query_items = llm_output.strip().split(" ")
        if query_items[0] == "ffmpeg":
            query_items = query_items[1:]

        full_cmd = ["ffmpeg", "-loglevel", "error", "-y"] + query_items
        print(f'\n\nrunning command: {" ".join(full_cmd)}')
        output = subprocess.run(full_cmd, capture_output=True),
        
        if type(output) == tuple:
            output = output[0]

        stdout = output.stdout
        stderr = output.stderr
        returncode = output.returncode
        print(f'[ffmpeg exited with {returncode}]')
        if stdout:
            print(f'\n{stdout.decode()}')
        if stderr:
            return f'[there was an error]\n{stderr.decode()}'

        return stdout.decode()

    async def _arun(
            self, 
            query: str, 
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        ) -> str:
            """Use the tool asynchronously."""
            async def async_run(cmd):
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await proc.communicate()

                print(f'[{cmd!r} exited with {proc.returncode}]')
                if stdout:
                    print(f'[stdout]\n{stdout.decode()}')
                if stderr:
                    print(f'[stderr]\n{stderr.decode()}')
                    raise Exception(f'[stderr]\n{stderr.decode()}')

                return stdout

            query_items = query.split(" ")
            if query_items[0] == "ffmpeg":
                query_items = query_items[1:]

            full_cmd = ["ffmpeg"] + query_items
            await async_run(full_cmd),

            return "output of this was saved to ./output.wav"
