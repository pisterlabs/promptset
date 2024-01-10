from openai import OpenAI

API_KEY = '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'

client = OpenAI(api_key=API_KEY)
#asst_SMTWUrT6Z9PcyDG0tYmVbMAy
# assistant = client.beta.assistants.create(
#     name="Math Tutor2",
#     instructions="You are a personal math tutor. Write and run code to answer math questions.", #지시사항
#     tools=[{"type": "code_interpreter"}], # 코드인터프리터, 리트리버
#     model="gpt-4-1106-preview" # 모델지정
# )
# print(assistant)
#Thread(id='thread_lqLzg67fNnm9sbxEVMTRZaBU
#thread = client.beta.threads.create()
#print(thread)

#ThreadMessage(id='msg_oRk0d95GKcRng4Lm4QdXamG2Y'
# message = client.beta.threads.messages.create(
#     thread_id="thread_lqLzg67fNnm9sbxEVMTRZaBU",
#     role="user",
#     content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
# )
# print(message)

# run = client.beta.threads.runs.create(
#   thread_id="thread_lqLzg67fNnm9sbxEVMTRZaBU",
#   assistant_id="asst_SMTWUrT6Z9PcyDG0tYmVbMAy",
#   instructions="Please address the user as Jane Doe. The user has a premium account."
# )
# print(run)

# run = client.beta.threads.runs.retrieve(
#   thread_id="thread_lqLzg67fNnm9sbxEVMTRZaBU",
#   run_id="run_a1xbDJcXxoYhySVgAb8gbC3d"
# )

messages = client.beta.threads.messages.list(
  thread_id="thread_lqLzg67fNnm9sbxEVMTRZaBU"
)
print("답변 : " + messages.data[3].content[0].text.value)