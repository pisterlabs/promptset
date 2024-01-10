import json
import langchain

from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from .helpers.utils import (
    instruction_response,
    output_format_checker,
    generate_full_prompt,
)


def get_launch_date(document_name: str) -> str:
    # Lauch date
    with open(document_name) as f:
        data = json.load(f)
    launch_date = None
    for key in data["key-value_pairs"]:
        if "trade date" in key.lower():
            launch_date = data["key-value_pairs"][key]
            # print(launch_date)

        elif "launch date" in key.lower():
            launch_date = data["key-value_pairs"][key]
            # print(launch_date)

        else:
            pass

    if launch_date is not None:
        prompt = f""" Given a list,  I want you to output the Launch Date into this format: dd/mm/yyyy. I just want the date, not the rest of the text. Dont create funcitons and staff, 
        JUST RETURN ME THE DATE. Take a deep breah and relax. Think step by step. Follow this examples:

        ##Example:
        ['February 22th, 2023', 0.855]
        ##Output:
        '22/02/2023'

        ##Example:
        ['March 24th, 2013', 0.855]
        ##Output:
        '24/03/2013'

        ##Example:
        {launch_date}
        ##Output( justthe date in dd/mm/yyyy format):"""

        launch_date = instruction_response(prompt).strip()

        # print("Launch date:")
        # print(launch_date)
        return launch_date
    else:
        documents = data["full_text"]
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0.2, separator=" "
        )
        texts = text_splitter.split_text(documents)
        embeddings = OpenAIEmbeddings(
        )
        db = FAISS.from_texts(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        launch_date = retriever.get_relevant_documents(
            "Launch Date",
        )

        prompt2 = """

        Extract the launch date of the term sheet.
        You need to extract the launch date of the term sheet.
        Given the following context:  {context} , extract the launch date of the term sheet.
        Read carefully, my life depends on it.
        Your response format example: dd/mm/yyyy.
        """
        launch_date = instruction_response(prompt2.format(context=launch_date)).strip()
        return launch_date
