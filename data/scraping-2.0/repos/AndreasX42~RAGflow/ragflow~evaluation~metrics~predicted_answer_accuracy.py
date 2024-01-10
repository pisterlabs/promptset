import numpy as np

from langchain.evaluation.qa import QAEvalChain
from langchain.schema.language_model import BaseLanguageModel

from ragflow.evaluation.utils import extract_llm_metric
from ragflow.commons.configurations import CVGradeAnswerPrompt
from ragflow.commons.prompts import (
    GRADE_ANSWER_PROMPT_5_CATEGORIES_5_GRADES_ZERO_SHOT,
    GRADE_ANSWER_PROMPT_3_CATEGORIES_4_GRADES_FEW_SHOT,
)

import logging

logger = logging.getLogger(__name__)


def grade_predicted_answer(
    label_dataset: list[dict],
    predicted_answers: list[str],
    grader_llm: BaseLanguageModel,
    grade_answer_prompt: CVGradeAnswerPrompt,
) -> tuple[float, float, float]:
    """_summary_Using a QAEvalChain and a LLM for grading we calculate a grade for the predicted answers from the query and the retrieved document chunks. We provide zero shot and few shot prompts and get the LLM to provide a score for correctness, comprehensiveness and readability of the predicted answers.

    Args:
        label_dataset (list[dict]): The evaluation ground truth dataset of QA pairs
        predicted_answers (list[str]): The predicted answers
        grader_llm (BaseLanguageModel): The LLM used for grading
        grade_answer_prompt (CVGradeAnswerPrompt): The prompt used in the QAEvalChain

    Returns:
        tuple[float, float, float]: Returns average scores for correctness, comprehensiveness and readability
    """

    logger.info("Grading generated answers.")

    if grade_answer_prompt == CVGradeAnswerPrompt.ZERO_SHOT:
        prompt, MAX_GRADE = GRADE_ANSWER_PROMPT_5_CATEGORIES_5_GRADES_ZERO_SHOT, 5
    elif grade_answer_prompt == CVGradeAnswerPrompt.FEW_SHOT:
        prompt, MAX_GRADE = GRADE_ANSWER_PROMPT_3_CATEGORIES_4_GRADES_FEW_SHOT, 3
    else:
        prompt, MAX_GRADE = None, 1

    # Note: GPT-4 grader is advised by OAI model_name="gpt-4"
    eval_chain = QAEvalChain.from_llm(llm=grader_llm, prompt=prompt, verbose=False)

    outputs = eval_chain.evaluate(
        label_dataset,
        predicted_answers,
        question_key="question",
        prediction_key="result",
    )

    # vectorize the function for efficiency
    v_extract_llm_metric = np.vectorize(extract_llm_metric)

    outputs = np.array([output["results"] for output in outputs])

    correctness = v_extract_llm_metric(outputs, "CORRECTNESS")
    comprehensiveness = v_extract_llm_metric(outputs, "COMPREHENSIVENESS")
    readability = v_extract_llm_metric(outputs, "READABILITY")

    return (
        np.nanmean(correctness) / MAX_GRADE,
        np.nanmean(comprehensiveness) / MAX_GRADE,
        np.nanmean(readability) / MAX_GRADE,
    )
