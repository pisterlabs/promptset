# from langchain.document_loaders import DirectoryLoader
# import glob
# import asyncio

# async def chunk_files():
#     print("here")
#     loader = DirectoryLoader('../', glob="**/*.txt", show_progress=True, use_multithreading=True)
#     docs = await loader.load()
#     print(len(docs))
#     print("done")

# asyncio.run(chunk_files())

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun

llm = OpenAI(temperature=0)
tools = load_tools(["requests_all", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

code = """
import * as vscode from "vscode";
import { getOpenAIApi } from "./api";

export const onDocumentSave = async (document: vscode.TextDocument) => {
  const api = getOpenAIApi();
  vscode.window.showInformationMessage("Saved!");
  const filename = document.fileName;
  const languageId = document.languageId;
  const text = document.getText();
};
"""

question = f"You are a master coder. What are some ways you would improve the following code for code quality? {code}"
agent.run(question)