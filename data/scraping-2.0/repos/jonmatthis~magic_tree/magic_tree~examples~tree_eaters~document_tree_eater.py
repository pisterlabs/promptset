import asyncio
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from pathlib import Path
from typing import Union

from magic_tree.controller.builders.directory_tree_builder import DirectoryTreeBuilder
from magic_tree.examples.tree_eaters.helpers.document_llm_chains import create_component_summary_chain, \
    create_global_summary_chain

load_dotenv()


async def document_tree_eater(document_root_path: Union[Path, str]):
    output_text = ""

    component_summary_chain = create_component_summary_chain()
    global_summary_chain = create_global_summary_chain()
    input_texts = []
    all_text = []
    subfolders_to_skip = ["bibliography", "boilerplate", "figures", "header"]
    for path in Path(document_root_path).rglob("*"):
        if any(subfolder in str(path).lower() for subfolder in subfolders_to_skip):
            continue
        if path.is_file():
            if path.suffix == ".tex":
                print(f"=============================================================\n"
                      f"Processing {path}")
                file_content = path.read_text()

                text_to_analyze = (f"{str(path)}\n"
                                   f"{file_content}")
                all_text.append(text_to_analyze)
                input_texts.append({"text": text_to_analyze})

    all_summaries = await component_summary_chain.abatch(input_texts)
    for summary in all_summaries:
        file_summary = (f"+++++++++++++++++++++++++++++++++++\n\n"
                        f"{summary.content}\n\n"
                        f"-----------------------------------\n\n")
        print(file_summary)
        output_text += file_summary

    print(output_text)

    global_summary = global_summary_chain.invoke({"text": "\n".join(all_text)}).content

    output_text += (f"=============================================================\n\n"
                    f"=============================================================\n\n"
                    f"___ \n > Global Summary \n {global_summary}\n\n")

    document_tree = DirectoryTreeBuilder.from_path(path=document_root_path)

    document_tree_string = f"```\n\n{document_tree.print()}\n\n```\n\n"
    print(document_tree_string)
    output_text += document_tree_string

    with open("document_summary.md", "w", encoding="utf-8") as file:
        file.write(output_text)


if __name__ == "__main__":
    document_root_path_in = Path(__file__).parent.parent.parent / "document"
    asyncio.run(document_tree_eater(document_root_path=document_root_path_in))
