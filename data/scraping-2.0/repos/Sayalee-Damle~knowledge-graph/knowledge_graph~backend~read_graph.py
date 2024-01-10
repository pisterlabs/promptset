import os
from pathlib import Path
from uuid import uuid4
from langchain.chains import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from knowledge_graph.configuration.toml_support import read_prompts_toml
from knowledge_graph.configuration.config import cfg

prompts = read_prompts_toml()


def read_graph_gefx(graph_path: Path):
    f = open(graph_path, "r")
    graph_f = f.read()
    return graph_f


def prompt_factory() -> ChatPromptTemplate:
    """Prompt Factory to derive graph description"""
    section = prompts["graph_desc"]
    human_message = section["human_message"]
    prompt_msgs = [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=section["system_message"], input_variables=[]
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=human_message,
                input_variables=["graph"],
            )
        ),
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


async def return_description(g_path):
    """The XML file for each graph is analyzed and summarized"""
    prompt = prompt_factory()
    graph = read_graph_gefx(g_path)
    chain = LLMChain(llm=cfg.llm, prompt=prompt)
    desc = await chain.arun({"graph": graph})
    return desc


async def save_description(sub_path: Path, path_desc: Path):
    """summary of each subgraph is put into a file"""
    desc = await return_description(sub_path)
    print(path_desc)
    with open(path_desc, "a+") as f:
        f.write("===========================")
        f.write(desc)
        f.write("\n " + "\n")




if __name__ == "__main__":
    """file = Path("/tmp/subgraphs/subgraph_1.gefx")
    #desc = read_graph_gefx(file)
    return_description(file)"""
    folder_path = cfg.save_fig_path
    
