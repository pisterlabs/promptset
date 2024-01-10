import numpy as np

from langchain.evaluation.qa import QAEvalChain
from langchain.schema.language_model import BaseLanguageModel

from ragflow.commons.prompts import GRADE_RETRIEVER_PROMPT
from ragflow.evaluation.utils import extract_llm_metric
from ragflow.commons.configurations import CVGradeRetrieverPrompt

import logging

logger = logging.getLogger(__name__)


def grade_retriever(
    label_dataset: list[dict],
    retrieved_docs: list[str],
    grader_llm: BaseLanguageModel,
    grade_docs_prompt: CVGradeRetrieverPrompt,
) -> float:
    """Using LangChains QAEvalChain we use a LLM as grader to get a metric how well the found document chunks from the vectorstore provide the answer for the label QA pairs. Per QA pair the LLM has to rate the retrieved document chunks with 0 or 1 depending if the question can be answered with the provided document chunks.

    Args:
        label_dataset (list[dict]): The evaluation ground truth dataset of QA pairs
        retrieved_docs (list[str]): The retrieved document chunks of each QA pair
        grader_llm (BaseLanguageModel): The LLM grading the document chunks
        grade_docs_prompt (CVGradeRetrieverPrompt): The type of prompt used in the QAEvalChain

    Returns:
        float: The average score of all QA pairs
    """

    logger.info("Grading retrieved document chunks.")

    # TODO: Provide more Prompts and more detailed prompt engineering
    if grade_docs_prompt == CVGradeRetrieverPrompt.DEFAULT:
        prompt = GRADE_RETRIEVER_PROMPT
    else:
        prompt = GRADE_RETRIEVER_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt)

    outputs = eval_chain.evaluate(
        label_dataset,
        retrieved_docs,
        question_key="question",
        answer_key="answer",
        prediction_key="retrieved_docs",
    )

    # vectorize the function for efficiency
    v_extract_llm_metric = np.vectorize(extract_llm_metric)

    outputs = np.array([output["results"] for output in outputs])
    retrieval_accuracy = v_extract_llm_metric(outputs, "GRADE")

    return np.nanmean(retrieval_accuracy)
