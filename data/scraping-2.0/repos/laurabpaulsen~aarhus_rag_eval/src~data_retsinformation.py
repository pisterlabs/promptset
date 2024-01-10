#!/usr/bin/env python3

import re
import logging

from datasets import load_dataset
from langchain_core.documents.base import Document


split_rules = re.compile(
    "\n((?:§{1,2}[ \t]*\d+-?\d*[ \t]*\w*)" + "|" +
    "(?:Kapitel[ \t]*\d+[ \t]*\w*)" + "|" +
    "(?:Afsnit[ \t]*[IVXLMCD]+))(?:[ \t]*\.*[ \t]*)")


def load_retsinformation(paragraph = True) -> list[Document]:
    """
    Download retsinformation (part of gigaword) and pick out sundhedsloven and serviceloven.
    Split each law into langchain Documents by either paragraph § or Chapter (Kapitel)
    Return a list of Documents with both serviceloven and sundhedsloven
    """
    logging.info("Filtering gigaword to select only retsinformation")
    gigaword = load_dataset("DDSC/partial-danish-gigaword-no-twitter")
    # retsinformation = gigaword.filter(lambda x: x['source'] == "retsinformationdk")

    ### Sundhedsloven
    sundhedsloven = gigaword.filter(lambda x: x['uri'] == 'https://www.retsinformation.dk/Forms/R0710.aspx?id=203757')
    if paragraph:
        sundhedsloven = split_lov_by_paragraph(sundhedsloven['train'][0]['text'], "Sundhedsloven")
    else:
        sundhedsloven = split_lov_by_kapitel(sundhedsloven['train'][0]['text'], "Sundhedsloven")
    sundhedsloven = list(filter(lambda x: "XX" not in x.metadata['afsnit'], sundhedsloven)) # Afsnit 20 (XX) is not relevant I think

    ### Serviceloven
    serviceloven = gigaword.filter(lambda x: x['uri'] == 'https://www.retsinformation.dk/Forms/R0710.aspx?id=202239')
    if paragraph:
        serviceloven = split_lov_by_paragraph(serviceloven['train'][0]['text'].replace(u'\xa0', u' '), "Serviceloven")
    else:
        serviceloven = split_lov_by_kapitel(serviceloven['train'][0]['text'].replace(u'\xa0', u' '), "Serviceloven")
    serviceloven = list(filter(lambda x: "Kapitel 34" not in x.metadata['kapitel'], serviceloven)) # Kapitel 34 is not relevant I think

    return sundhedsloven + serviceloven


def split_lov_by_kapitel(text: str, source: str) -> list[Document]:

    ## split by 3 different headlines defined in split_rules:
    ## afsnit XX, kapitel yy, § zz
    raw_list = split_rules.split(text)
    ## build a list of ("§ 1", "text"), ("kapitel 1", "text"), ("afsnit 1", "text") pairs
    combined_list = [raw_list[i:i+2] for i in range(1, len(raw_list), 2)]

    afsnit=""
    kapitel=""
    paragraph_list = ""
    for header, elem in combined_list:

        # build per-kapitel representation
        if header[:7] == "Kapitel":
            if paragraph_list:
                # if we have some text to store (ie we're not at kapitel 1 or in index)
                yield Document(page_content = f"""{afsnit}\n{kapitel}\n{paragraph_list}""",
                               metadata = {'source': source, 'kapitel': kapitel, 'afsnit': afsnit, 'title': f"{source}: {kapitel}"})
                paragraph_list = ""

            kapitel = f"{header}: {elem.strip()}"

        elif header[:6] == "Afsnit":
            afsnit = f"{header}: {elem.strip()}"
        elif header[0] == "§":
            paragraph_list += f"{header}: {elem.strip()}\n"
        else:
            logger.debug(f"Text segment categorizatioon failed for {header}: {elem}")

    # finish off last kapitel
    yield Document(page_content = f"""{kapitel}\n{afsnit}\n{paragraph_list}""",
                   metadata = {'source': source, 'kapitel': kapitel, 'afsnit': afsnit, 'title': f"{source}: {kapitel}"})


def split_lov_by_paragraph(text: str, source: str) -> list[Document]:

    ## same logic as split_lov_by_kapitel

    raw_list = split_rules.split(text)
    combined_list = [raw_list[i:i+2] for i in range(1, len(raw_list), 2)]

    afsnit=""
    kapitel=""
    for header, elem in combined_list:

        # build per-paragraph representation
        if header[:7] == "Kapitel":
            kapitel = f"{header}: {elem.strip()}"
        elif header[:6] == "Afsnit":
            afsnit = f"{header}: {elem.strip()}"
        elif header[0] == "§":
            yield Document(page_content = f"""{afsnit}\n{kapitel}\n{header}: {elem.strip()}""",
                           metadata = {'source': source, 'kapitel': kapitel, 'afsnit': afsnit, 'paragraph': header, 'title': f"{source}: {header}"})
        else:
            logger.debug(f"Text segment categorizatioon failed for {header}: {elem}")



if __name__ == '__main__':
    ri = load_retsinformation()
    assert len(ri) == 608
    assert all(['source' in x.metadata for x in ri])

    ri2 = load_retsinformation(paragraph = False)
    assert len(ri2) == 127
    assert all(['source' in x.metadata for x in ri2])
