from utils import *
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.llms import HuggingFacePipeline
from prompts import QUESTION_PROMPT
from langchain import LLMChain, PromptTemplate
import os
import pandas as pd
import numpy as np


class RefineAnswer(BaseModel):
    answer: str = Field(..., description="Answer to the question")


def load_model(llm_model="HuggingFaceH4/zephyr-7b-beta"):
    """load the model"""
    llm_model = HuggingFacePipeline.from_model_id(
        model_id=llm_model, task="text-generation", pipeline_kwargs={"max_new_tokens": 30}
    )

    return llm_model


def run_basic_extractor(filedir, queries, headers, llm, add_previous=True, chunk_size=1000):
    """run basic extractor on all files in filedir.
    Returns a dataframe with the answers from each file."""

    refiner = PydanticOutputParser(pydantic_object=RefineAnswer)

    prompt = PromptTemplate(
        template=QUESTION_PROMPT,
        input_variables=["text", "question", "previous_qa"],
        partial_variables={"format_instructions": refiner.get_format_instructions()},
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    rows = []
    for file in os.listdir(filedir):
        filename = os.path.join(filedir, file)
        print(filename)
        previous = ""
        for query in queries:
            doc_chunks = search_docs(
                filename, query, chunk_size=chunk_size, chunk_overlap=chunk_size * 0.1
            )

            evidence = ""

            for doc in doc_chunks:
                chunk = doc.page_content
                evidence += chunk

            unformatted_answer = llm_chain.run(
                {"text": evidence, "question": query, "previous_qa": previous}
            )

            try:
                parsed_answer = refiner.parse(unformatted_answer).dict()
                rows.append(parsed_answer["answer"])

            except:
                rows.append(unformatted_answer)

            if add_previous:
                previous += f"Q: {query} \n A: {unformatted_answer} \n\n"

    rows = np.array(rows).reshape(-1, len(queries))
    df = pd.DataFrame(rows, columns=headers)
    df.insert(0, "filename", os.listdir(filedir))

    return df
