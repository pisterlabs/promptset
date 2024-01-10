import os
from openai import OpenAI
from openai.types.beta import Assistant, AssistantDeleted
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv('OPENAI_KEY')

client = OpenAI(api_key=openai_key)

# RetrieveAssistant
 
graph_wiz = client.beta.assistants.retrieve('asst_2VF4iqoOjt4WF1hyR2TIFqWG')
# print(graph_wiz)

# RetrieveThread

my_thread = client.beta.threads.retrieve('thread_eDgqzOkf5Uh5kTi5TVq5r2bZ')
# print(my_thread)

my_thread_better = client.beta.threads.retrieve('thread_cmqq7dx1GsXEhjLtgWjmpa7T')
# print(my_thread_better)

thread_message_test = client.beta.threads.messages.create(
    'thread_cmqq7dx1GsXEhjLtgWjmpa7T',
    role='user',
    content="Return the GraphQL Query that would answer: Who was on the Houston Rockets Roster in 2016?",
)

# print(thread_message_test)



message = client.beta.threads.messages.retrieve(
    message_id='msg_3kRqAKf354y0fVv7946GitDV',
    thread_id='thread_cmqq7dx1GsXEhjLtgWjmpa7T',
)

# print(message)

thread_messages = client.beta.threads.messages.list('thread_cmqq7dx1GsXEhjLtgWjmpa7T')
# print(thread_messages)


run = client.beta.threads.runs.create(
    thread_id='thread_cmqq7dx1GsXEhjLtgWjmpa7T',
    assistant_id='asst_2VF4iqoOjt4WF1hyR2TIFqWG',
    instructions='Please return the GraphQL query that would return the 2016 OKC roster'
)

print(run)

run_status = client.beta.threads.runs.retrieve(
  thread_id='thread_cmqq7dx1GsXEhjLtgWjmpa7T',
  run_id='run_o7Oa2HKzgvE0tHORK8cQLmtr'
)

# print(run_status)

messages = client.beta.threads.messages.list(
  thread_id='thread_cmqq7dx1GsXEhjLtgWjmpa7T'
)

# print(messages)