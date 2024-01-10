#wget -r -A.html -P scraped_docs_data https://diataxis.fr

import asyncio
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from magic_tree.controller.builders.directory_tree_builder import DirectoryTreeBuilder

load_dotenv()

SUMMARY_PROMPT = """
    You are a bot assistant for an open source software project. 
    We are working on giving the documentation for this project a complete overhaul.
    This will be an ongoing, iterative process.
    
    We are assessing each document as it currently exists to be reworked so that it fits into the Diataxis Documentation framework.
    The Diataxis Documentation framework can be summarized as follows:
        * The framework provides a systematic approach to create, organize and maintain technical documentation.
        * The goal is pragmatic improvement.
        * The framework divides documentation into 4 types based on 2 axes:
            1 - Theory vs Practice
            2 - Acquisition vs Application
        * The 4 types of Documentation:
            1 - Tutorials - Learning-oriented guides that provide lessons to teach users basic skills. Help users get started.
            2 - How-To Guides - Task-oriented guides that provide steps to accomplish specific goals. Help users solve problems.
            3 - Reference - Information-oriented technical descriptions of the product. Help users find factual information.
            4 - Explanation - Understanding-oriented discussion to provide context and illuminate concepts. Help users gain deeper knowledge.
        * Each type serves a distinct user need at different points in their journey using the product.
        * Keeping the types clearly separated improves quality by ensuring docs fit user needs.
        * Start small, assess and improve one part at a time to steadily enhance overall documentation.
        
    Your job as a bot assistant is to:
        * At the top of each document, add text identifying it so that it fits into the Diataxis framework
        * Summarize each document so that it considers the Diataxis Documentation framework.
 
 DOCUMENT DRAFT TEXT: 
 
 {text}

"""


def create_component_summary_chain():
    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    model = ChatOpenAI(temperature=0,
                       model_name="gpt-3.5-turbo-16k")
    chain = prompt | model
    return chain


GLOBAL_SUMMARY_PROMPT = """
    You are a bot assistant for an open source software project. 
    We are working on giving the documentation for this project a complete overhaul.
    This will be an ongoing, iterative process.
    
    We are assessing each document as it currently exists to be reworked so that it fits into the Diataxis Documentation framework.
    The Diataxis Documentation framework can be summarized as follows:
        * The framework provides a systematic approach to create, organize and maintain technical documentation.
        * The goal is pragmatic improvement.
        * The framework divides documentation into 4 types based on 2 axes:
            1 - Theory vs Practice
            2 - Acquisition vs Application
        * The 4 types of Documentation:
            1 - Tutorials - Learning-oriented guides that provide lessons to teach users basic skills. Help users get started.
            2 - How-To Guides - Task-oriented guides that provide steps to accomplish specific goals. Help users solve problems.
            3 - Reference - Information-oriented technical descriptions of the product. Help users find factual information.
            4 - Explanation - Understanding-oriented discussion to provide context and illuminate concepts. Help users gain deeper knowledge.
        * Each type serves a distinct user need at different points in their journey using the product.
        * Keeping the types clearly separated improves quality by ensuring docs fit user needs.
        * Start small, assess and improve one part at a time to steadily enhance overall documentation.
        
    Approach this as a second pass. Previously, we summarized each document individually in the repository. 
    For this pass, your job as a bot assistant is to:
        * I need you to combine the summaries of each file into a global report of the current state of the documentation. This should include:
            1. Identification of gaps that would complete an approach using the Diataxis Documentation framework.
            2. A prioritized list of what exists and where the gaps should most be addressed. 
             
 CURRENT PROPOSAL TEXT: 
 
{text}
"""


def create_global_summary_chain():
    prompt = ChatPromptTemplate.from_template(GLOBAL_SUMMARY_PROMPT)

    model = ChatAnthropic(temperature=0, max_tokens=4000)
    chain = prompt | model
    return chain


async def auto_summary(document_root_path: Union[Path, str],
                       extension:str):

    if not extension.startswith("."):
        raise Exception("Extension must start with `. `")

    output_text = ""

    component_summary_chain = create_component_summary_chain()
    global_summary_chain = create_global_summary_chain()
    input_texts = []
    all_text = []
    paths =[]
    subfolders_to_skip = ["_static", "_templates","images"]
    for path in Path(document_root_path).rglob(f"*{extension}"):
        if any(subfolder in str(path).lower() for subfolder in subfolders_to_skip):
            continue
        if path.is_file():
            try:
                file_content = path.read_text()
                print(f"=============================================================\n"
                      f"Processing {path}")
                paths.append(path)
                text_to_analyze = (f"{str(path)}\n"
                                   f"{file_content}")
                all_text.append(text_to_analyze)
                input_texts.append({"text": text_to_analyze})
            except Exception as e:
                print(f"Unable to process {path} because `{e}`...")



    all_summaries = await component_summary_chain.abatch(input_texts)
    for path, summary in zip(paths, all_summaries):
        file_summary = (f"+++++++++++++++++++++++++++++++++++\n\n"
                        f"INDIVIDUAL FILE PATH - {path}\n\n"
                        f"{summary.content}\n\n"
                        f"-----------------------------------\n\n")
        print(file_summary)
        output_text += file_summary

    print(output_text)

    global_summary = global_summary_chain.invoke({"text": "\n".join(all_text)}).content

    output_text += (f"=============================================================\n\n"
                    f"=============================================================\n\n"
                    f"\n\n GLOBAL SUMMARY OF SUMMARIES \n \n"
                    f"=============================================================\n\n"
                    f"{global_summary}\n\n")

    # document_tree = DirectoryTreeBuilder.from_path(path=document_root_path)
    #
    # document_tree_string = f"```\n\n{document_tree.print()}\n\n```\n\n"
    # print(document_tree_string)
    # output_text += document_tree_string


    with open("document_summary-diataxis0.md", "w", encoding="utf-8") as file:
        file.write(output_text)


if __name__ == "__main__":
    # document_root_path_in = Path("./diataxis-documentation-framework-repo")
    document_root_path_in = Path("./scraped_docs_data/documentation/docs")
    asyncio.run(auto_summary(document_root_path=document_root_path_in, extension=".md"))
