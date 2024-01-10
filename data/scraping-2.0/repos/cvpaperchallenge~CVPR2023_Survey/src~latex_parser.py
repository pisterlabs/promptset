import re
from typing import Any, Dict, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter


def parse_latex_text(latex_document: str) -> dict[str, Any]:
    # Remove metadata contents before abstract
    _, contents = latex_document.split("\\begin{abstract}")

    # Extract abstract
    abstract, contents_wo_abstract = contents.split("\n\\end{abstract}")

    # Split sections
    raw_section_list = contents_wo_abstract.lstrip("\n").split("\\section{")
    section_list = []
    section_id = 1
    for i, each_section in enumerate(raw_section_list):
        if i == 0:
            continue
        section_title, raw_section_text = each_section.split("}\n", 1)
        section_text = raw_section_text.lstrip("\n")
        section_text = simple_figure_table_remover(section_text)
        section_dict = {
            "section_id": section_id,
            "section_title": section_title,
            "section_text": section_text,
        }
        section_list.append(section_dict)
        section_id += 1

    # Split subsections
    for each_section_dict in section_list:
        raw_subsection_list = each_section_dict["section_text"].split("\\subsection{")
        # Go into next section if there is no subsection
        if len(raw_subsection_list) == 1:
            each_section_dict["subsection_list"] = []
            continue
        subsection_list = []
        subsection_id = 0
        for each_subsection in raw_subsection_list:
            # If there is no text between \section{} and \subsection{}, skip it
            if len(each_subsection) == 0:
                subsection_id += 1
                each_section_dict["section_text"] = ""
                continue
            # If there is text between \section{} and \subsection{}, update section_text
            if subsection_id == 0 and len(each_subsection) != 0:
                each_section_dict["section_text"] = each_subsection
                subsection_id += 1
                continue

            subsection_title, raw_subsection_text = each_subsection.split("}\n", 1)
            subsection_text = raw_subsection_text.lstrip("\n")
            subsection_dict = {
                "subsection_id": subsection_id,
                "subsection_title": subsection_title,
                "subsection_text": subsection_text,
            }
            subsection_list.append(subsection_dict)
            subsection_id += 1

        each_section_dict["subsection_list"] = subsection_list

    parsed_document = {
        "abstract": abstract,
        "section": section_list,
    }

    return parsed_document


def simple_figure_table_remover(text: str) -> str:
    wo_table_text = re.sub(
        r"\\begin{tabular}(.*?)\\end{tabular}", "", text, flags=re.DOTALL
    )
    wo_fig_table_text = re.sub(r"!\[\]\((.*?)\)\n", "", wo_table_text, flags=re.DOTALL)
    return wo_fig_table_text


def structure_latex_documents(
    parsed_paper: Dict,
    text_splitter: TextSplitter,
    abstract_text: Optional[str] = None,
) -> List[Document]:
    # If full abstract is provided, use it instead of parsed one.
    documents = [
        Document(
            page_content=abstract_text if abstract_text else parsed_paper["abstract"],
            metadata={"section": "abstract"},
        )
    ]

    # Loop over section.
    for each_section in parsed_paper["section"]:
        section_title = each_section["section_title"]
        if section_title == "References":
            continue

        section_id = each_section["section_id"]
        section_text = each_section["section_text"]
        if section_text:
            metadata = {"section_id": f"{section_id}", "section": f"{section_title}"}
            for each_section_text in text_splitter.split_text(section_text):
                documents.append(
                    Document(page_content=each_section_text, metadata=metadata)
                )

        # Loop over subsection.
        for each_subsection in each_section["subsection_list"]:
            subsection_title = each_subsection["subsection_title"]
            subsection_id = each_subsection["subsection_id"]
            subsection_text = each_subsection["subsection_text"]
            metadata = {
                "section_id": f"{section_id}.{subsection_id}",
                "section": f"{section_title}/{subsection_title}",
            }
            for each_subsection_text in text_splitter.split_text(subsection_text):
                documents.append(
                    Document(page_content=each_subsection_text, metadata=metadata)
                )

    return documents
