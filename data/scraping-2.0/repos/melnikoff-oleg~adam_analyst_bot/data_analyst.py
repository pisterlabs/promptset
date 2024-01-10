import io
import logging
import traceback
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import plotly as plotly
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI

from agent import BaseMinion
from common_prompts import TableDescriptionPrompt
from custom_python_ast import CustomPythonAstREPLTool

from settings import Settings


def preparation(
    path: str,
    build_plots: False,
    user_data_description: str,
):
    sheet_name = "Sheet1"
    file_extension = pathlib.Path(path).suffix
    if file_extension == ".XLSX":
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif file_extension == ".json":
        df = pd.read_json(path)
    elif file_extension == ".csv":
        df = pd.read_csv(path)
    else:
        raise Exception("Unknown file extension")

    df_head = df.head()
    df_info = io.StringIO()
    df.info(buf=df_info)

    settings = Settings()

    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4",  # gpt-3.5-turbo
        openai_api_key=settings.OPENAI_API_TOKEN,
    )

    python_tool = CustomPythonAstREPLTool(
        locals={"df": df, "python": None, "python_repl_ast": None},
        globals={"pd": pd, "np": np, "sns": sns, "plotly": plotly},
    )

    prompt = TableDescriptionPrompt(
        user_data_description,
        build_plots=build_plots,
    )

    ag = BaseMinion(
        base_prompt=prompt.__str__(),
        available_tools=[
            Tool(
                name=python_tool.name,
                description=python_tool.description,
                func=python_tool._run,
            )
        ],
        model=llm,
        max_iterations=50,
    )
    return ag, df_head, df_info


logging.basicConfig(level=logging.INFO, filename="py_log.log", filemode="w")


def use_data_assistant(
    data_path: str,
    data_description: str,
    question: str,
) -> str:
    build_plots = False
    # for plot_keyword in [
    #     "график",
    #     "нарисовать",
    #     "нарисуй",
    #     "распределение",
    #     "изобра",
    #     "chart",
    #     "plot",
    #     "graph",
    #     "draw",
    # ]:
    #     if plot_keyword in question.lower():
    #         build_plots = True
    ag, df_head, df_info = preparation(
        path=data_path,
        build_plots=build_plots,
        user_data_description=data_description,
    )
    try:
        return f"Answer: {ag.run(input=question, df_head=df_head, df_info=df_info.getvalue())}"
    except Exception as e:
        return f"Failed with error: {traceback.format_exc()}"
