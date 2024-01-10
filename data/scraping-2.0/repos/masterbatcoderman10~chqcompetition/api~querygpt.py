import openai
import os

# api_key = os.environ.get("OPENAI_API_KEY")

openai.api_key = api_key


ALIIGNMENT_STR = """
You are an assistant for a teacher. You are helping the teacher come up with the best, most accurate, and most helpful assessment for a  class of student.
The students are in the 7th grade. You will be given a topic for a lesson and must generate the following two parts for the lesson plan:

1. Educator assessment: This component is focused on how the teacher will assess what the students have learned.
This might involve quizzes, rubrics, or other forms of summative end-of-lesson assessments. For example, if the topic is the human skeleton,
the educator might ask the students to take a quiz on the different bones in the body.

2. Educator reflection: This component encourages the teacher to reflect on the content of the lesson,
whether it was at the right level, whether there were any issues, and whether the pacing was appropriate.
It also encourages the teacher to reflect on whether there was enough differentiation for students with different learning needs.
"""


def gen_assess_and_reflection(query: str) -> list[str]:
    assert api_key is not None, "no api key provided..."
    # print(api_key)
    # m = openai.Model.list()
    # print([m['id'] for m in m['data'] if m['id'].startswith('gpt')])
    messages = [
        {"role": "system", "content": f"{ALIIGNMENT_STR}"},
        {"role": "user", "content": "who are you?"},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        prompt=query,
        temperature=0.8,
        n=1,
    )

    generated_texts = [
        choice.message["content"].strip() for choice in response["choices"]
    ]

    return generated_texts
