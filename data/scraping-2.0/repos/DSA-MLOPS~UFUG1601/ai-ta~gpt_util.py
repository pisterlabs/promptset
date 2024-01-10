import openai
import os
from good_prompt import SYSTEM_PROMPT, get_prompt

# Set openai.api_key to the OPENAI environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_MAX_TOKEN = 2048


def gpt_chat(msgs, call_back=None):
    stream = True if call_back else False
    result = ""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=msgs, stream=stream
    )
    if stream and call_back:
        for resp in response:
            if "content" in resp.choices[0].delta and resp.choices[0].delta.get(
                "content"
            ):
                result += resp.choices[0].delta.get("content")
                call_back(result)

        return result
    else:
        status_code = response["choices"][0]["finish_reason"]
        assert status_code == "stop", f"The status code was {status_code}."
        return response["choices"][0]["message"]["content"]


def gpt_grade(hw_desc, student_code, stdout, stderr, call_back=None):
    content = get_prompt(hw_desc, student_code, stdout, stderr)
    content = content[:OPENAI_MAX_TOKEN]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    return gpt_chat(msgs, call_back)


def gpt_mk_quiz(course_name: str = "Python Introduction", topic: str = "any", call_back_func=None) -> str:
    """
    Generates a GPT-3 prompt for creating a programming quiz.

    Parameters:
    course_name (str): The name of the course for the quiz.
    topic (str): The topic for the quiz. Defaults to "any".

    Returns:
    dict: A dictionary containing the system and user messages for the GPT-3 prompt.
    """
    if not course_name or not isinstance(course_name, str):
        raise ValueError("course_name must be a non-empty string.")

    if not topic or not isinstance(topic, str):
        raise ValueError("topic must be a non-empty string.")

    system_message = (
        "You are a very famous professor in a {course} programming class and you "
        "want to make a programming task for your students."
    ).format(course=course_name)

    user_message = (
        "# Quiz: {topic}\n\n"
        "Please write one programming task for the {topic} topic in the {course} programming course.\n"
        "It should include a title, tak description, and code output sample in markdown format. "
        "Create something that is educational yet enjoyable for students.\n"
        "Note: Only one problem should be provided."
    ).format(course=course_name, topic=topic)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return gpt_chat(messages, call_back_func)


if __name__ == "__main__":
    print(gpt_mk_quiz("Python introdution"))

    hw_desc = """
Write code to print out the followings:
*
**
***
****
***
**
*
"""

    student_code = """  
for i in range(1, 5):
    print("*" * i)
for i in range(5, 0, -1):
    print("*" * i)
"""
    stdout = """            
*   
**

***
****

***
**

*
"""
    stderr = ""
    print(gpt_grade(hw_desc, student_code, stdout, stderr))
