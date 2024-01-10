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
from agent.agent_network import ReActToolAgent, AgentNetWork
from agent.alm import ReActReflexionAgent
from agent.llm import GPT3_5LLM, Davinci003LLM
from agent.tools import ToolList
from utils.loadenv import Env
import openai
import json

env = Env()
llm = GPT3_5LLM(env.openai_key())
llm_ref = GPT3_5LLM(env.openai_key())
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
    R_ExecuteCodeTool(),
    R_CalculatorTool()
    # ReadLectureTool(""),
]
tool_list = ToolList()
for tool in tools:
    tool_list.registerTool(tool)

file = open("dataset/tableMWP/problems_test1k.json", "r")
obj: dict = json.loads(file.read())
data = []
for i in obj.values():
    data.append(i)
cai = False
i = 814
while i < 1000:
    try:
        d = data[i]
        query = d["question"] + "\n"
        query += "You need to read from this table to generate your answer:"
        query += d["table"] + "\n"
        ground_truth = ""
        if d["ans_type"].endswith("number"):
            def f(x): return abs(
                eval(x) - eval(d["answer"].replace(",", ""))) < 0.001
            ground_truth = eval(d["answer"].replace(",", ""))
        else:
            def f(x): return EM(x, d["answer"])
            ground_truth = d["answer"]
        tools[1].func = f
        react_agent = ReActReflexionAgent(llm, llm_ref, tool_list)
        react_agent.setRequest(query)
        llm.tokens = 0
        react_agent.steps(max_trials=2, max_steps=8)
        import os
        os.system(f"del ./logs/reflexion_tableMWP_{i}.log")
        react_agent.saveLog(
            f"reflexion_tableMWP_{i}", {
                "ground_truth": ground_truth, "token_use": llm.tokens}
        )
        i += 1
    except:
        continue
