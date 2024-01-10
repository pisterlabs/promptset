from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
MATH_TUTOR_ASST = os.environ['MATH_TUTOR_ASST']
MATH_TUTOR_THREAD_1 = os.environ['MATH_TUTOR_THREAD_1']
MATH_TUTOR_MSG = os.environ['MATH_TUTOR_MSG']
MATH_TUTOR_RUN = os.environ['MATH_TUTOR_RUN']

client = OpenAI(api_key=API_KEY)

# 어시스턴스
# assistant = client.beta.assistants.create(
#     name = "Math Tutor",
#     instructions="You are a personal math tutor. Write and run code to answer math questions",
#     tools = [{"type": "code_interpreter"}],
#     model = "gpt-3.5-turbo-16k"
# )
# print(assistant)

# 스레드
# thread = client.beta.threads.create()
# print(thread)

# 메세지
# message = client.beta.threads.messages.create(
#     thread_id = MATH_TUTOR_THREAD_1,
#     role = "user",
#     content = "I need to solve the equation `3x + 11 = 14`. Can you help me?"
# )
# print(message)

# 작동
# run = client.beta.threads.runs.create(
#     thread_id = MATH_TUTOR_THREAD_1,
#     assistant_id = MATH_TUTOR_ASST,
#     instructions ="Please address the user as Jane Doe. The user has a premium account."
# )
# print(run)

# run = client.beta.threads.runs.retrieve(
#     thread_id = MATH_TUTOR_THREAD_1,
#     run_id = MATH_TUTOR_RUN
# )
# print(run)

messages = client.beta.threads.messages.list(
    thread_id = MATH_TUTOR_THREAD_1
)

print(messages.data[0].content[0].text.value)
