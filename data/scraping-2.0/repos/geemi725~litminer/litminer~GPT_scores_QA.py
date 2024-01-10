from utils import *
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import os
import pandas as pd
import numpy as np


class ScoreChunk(BaseModel):
    relevance_score: int = Field(..., description="Relevance score of text summary")


class RefineAnswer(BaseModel):
    answer: str = Field(..., description="Answer to the question")
    literature: str = Field(..., description="literature evidence")
    score: int = Field(..., description="confidence score of the answer")


refine_parser = PydanticOutputParser(pydantic_object=RefineAnswer)
score_ans_parser = PydanticOutputParser(pydantic_object=ScoreChunk)


def run_extractor(file_dir, queries, **kwargs):
    """run basic extractor on all files in file_dir.
    Returns a dataframe with the answers from each file.
    kwargs: chunk_size, logfile, top_k, llm_model, headers"""

    chunk_size = kwargs.get("chunk_size", 1000)

    llm = ChatOpenAI(
        temperature=0.0, model_name=kwargs.get("llm_model", "gpt-4"), request_timeout=1000
    )

    logfile = kwargs.get("logfile", "log.txt")
    f = open(logfile, "w")

    headers = get_headers(queries, llm_model=kwargs.get("llm_model", "gpt-4"))
    answer_chain = set_answer_query(llm, parser=refine_parser)
    score_chain = set_score_chunks(llm=llm, parser=score_ans_parser)

    if kwargs.get("headers", None) is None:
        headers = get_headers(queries, llm_model=kwargs.get("llm_model", "gpt-4"))
    else:
        headers = kwargs.get("headers")

    rows = []
    for file in os.listdir(file_dir):
        print(file)
        f.write(file)
        filename = os.path.join(file_dir, file)
        previous = ""

        for i, query in enumerate(queries):
            doc_chunks = search_docs(
                filename,
                query,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size * 0.1,
                mmr_with_scores=False,
                top_k=kwargs.get("top_k", 10),
            )
            evidence = get_score_chunks(
                query, doc_chunks, parser=score_ans_parser, llmchain=score_chain
            )

            response = get_answer_query(
                evidence, query=query, previous=previous, llmchain=answer_chain
            )

            parsed = refine_parser.parse(response).dict()

            # round 2

            if parsed["answer"] == "not found":
                f.write(f"round 1 failed for query {i} \n")
                doc_chunks = search_docs(
                    filename,
                    query,
                    chunk_size=chunk_size / 2,
                    chunk_overlap=chunk_size / 20,
                    mmr_with_scores=False,
                    top_k=kwargs.get("top_k", 15),
                )

                evidence = get_score_chunks(
                    query, doc_chunks, parser=score_ans_parser, cutoff=7, llmchain=score_chain
                )

                response = get_answer_query(
                    evidence, query=query, previous=previous, llmchain=answer_chain
                )

                parsed = refine_parser.parse(response).dict()
                answer = parsed["answer"]
                f.write(f"round 2 answer {answer} \n")

            rows.append(parsed["answer"])
            rows.append(parsed["literature"])
            rows.append(parsed["score"])
            previous += f'Q: {headers[i*3]} \nA: {parsed["answer"]} \n\n'
        print(previous)

    f.close()
    rows = np.array(rows).reshape(-1, len(queries) * 3)
    df = pd.DataFrame(rows, columns=headers)
    df.insert(0, "filename", os.listdir(file_dir))

    return df
