from typing import List

from openai import OpenAI
from pydantic import BaseModel
from starlette.datastructures import State

LEADING_POSITIVE_FEEDBACK_TITLE: str = "Things done well"
LEADING_NEGATIVE_FEEDBACK_TITLE: str = "Things to improve upon"
NUM_SUGGESTIONS: int = 5


class QuestionAndAnswer(BaseModel):
    question: str
    answer: str


class Feedback(BaseModel):
    positive_feedback: List[str]
    negative_feedback: List[str]


class QuestionAnswerFeedback(BaseModel):
    question: str
    answer: str
    feedback: Feedback


async def question_answer_feedback(endpoint_input: QuestionAndAnswer, state: State) -> QuestionAnswerFeedback:
    suggestion_response = await get_completion(
        interview_question=endpoint_input.question,
        answer=endpoint_input.answer,
        open_api_key=state.api_keys["openai"]
    )
    feedback: Feedback = format_suggestions(suggestion_response)
    return QuestionAnswerFeedback(
        question=endpoint_input.question,
        answer=endpoint_input.answer,
        feedback=feedback
    )


async def get_completion(interview_question: str, answer: str, open_api_key: str, model="gpt-4"):
    client = OpenAI(api_key=open_api_key)

    prompt = "Given the following interview question:" + \
             "\n" + interview_question + \
             "\n" + "And the following answer:" + \
             "\n" + answer + \
             "\n" + f". Suggest a list of {NUM_SUGGESTIONS} things the answer did well, leading with the title: '{LEADING_POSITIVE_FEEDBACK_TITLE}'" + \
             "\n" + f"and {NUM_SUGGESTIONS} things the answer improve upon leading with the title: '{LEADING_NEGATIVE_FEEDBACK_TITLE}'."

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content


def format_suggestions(suggestion_response: str) -> Feedback:
    positive_feedback = []
    negative_feedback = []
    for i, line in enumerate(suggestion_response.split("\n")):
        if LEADING_POSITIVE_FEEDBACK_TITLE in line:
            positive_feedback.extend([l for l in suggestion_response.split("\n")[i+1:i+NUM_SUGGESTIONS+1]])
        elif LEADING_NEGATIVE_FEEDBACK_TITLE in line:
            negative_feedback.extend([l for l in suggestion_response.split("\n")[i+1:i+NUM_SUGGESTIONS+1]])
    return Feedback(positive_feedback=positive_feedback, negative_feedback=negative_feedback)