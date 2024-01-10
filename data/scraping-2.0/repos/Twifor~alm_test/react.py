from tkinter import E
from turtle import begin_fill
from agent.tools import ToolList
from examples.bmbtools.douban import (
    PrintDetailTool,
    NowPlayingOutFilterTool,
    ComingOutFilterTool,
)
from examples.hotpotQA.hotpotqa_tools import EM
import json
import random
from examples.bmbtools.weather import (
    ForcastWeatherTool,
    GetWeatherTool,
)
from examples.bmbtools.file_operation import WriteFileTool, ReadFileTool
from examples.bmbtools.wikipedia import (
    WikiPediaSearchTool,
    WikiLookUpTool,
    WikiPediaDisambiguationTool,
)
from examples.bmbtools.answer import AnswerTool, BeginTool
from examples.bmbtools.python import RunPythonTool
from examples.bmbtools.google_search import GoogleSearchTool, GoogleSearch2Tool
from examples.bmbtools.code_interpreter import ExecuteCodeTool
from examples.bmbtools.gradio import ImageCaptionTool, ImageToPromptTool, OCRTool
from examples.scienceQA.read_lecture import ReadLectureTool
from examples.scienceQA.rubbish_tools import *
from agent.alm import ReActAgent
from agent.llm import GPT3_5LLM, Davinci003LLM
from utils.loadenv import Env
import openai
import json

env = Env()
llm = GPT3_5LLM(env.openai_key())
tools = [
    BeginTool(),
    AnswerTool(lambda x: True),
    WikiPediaSearchTool(),
    WikiLookUpTool(),
    WikiPediaDisambiguationTool(),
    GoogleSearch2Tool(),
    OCRTool(),
    # ImageCaptionTool(),
    GoogleSearchTool(env.searper_key()),
    ExecuteCodeTool(),
    RunPythonTool(),
    R_OCRTool(),
    R_SearchTool(),
    R_UnknownTool(),
    R_LoopUpTool(),
    # ReadLectureTool(""),
    R_CalculatorTool(),
    R_ExecuteCodeTool(),
]
tool_list = ToolList()
for tool in tools:
    tool_list.registerTool(tool)

i = 670
while i < 1000:
    try:
        react_agent = ReActAgent(llm, tool_list)
        f = open(f"./dataset/math/{i}.json", "r")
        d = json.loads(f.read())
        query = d["problem"] + "\n"
        query += "Use Answer tool to submit your final answer. The answer should be an integer instead of an expression.\n"

        def f(x):
            return EM(x, d["answer"])

        ans = d["answer"]
        tools[1].func = lambda x: EM(x, ans)

        react_agent.setRequest(query)
        llm.tokens = 0
        react_agent.steps(max_steps=8)
        import os

        os.system(f"del ./logs/react_math_{i}.log")
        react_agent.saveLog(
            f"react_math_{i}", {"ground_truth": ans, "token_use": llm.tokens}
        )
        i += 1
    except Exception as e:
        print(e)
