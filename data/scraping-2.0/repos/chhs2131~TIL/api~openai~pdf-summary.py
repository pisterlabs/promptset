import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("CUSTOM_ENV_NAME")
)

# Upload a file with an "assistants" purpose
file = client.files.create(
  file=open("example_report.pdf", "rb"),
  purpose='assistants'
)

# Add the file to the assistant
assistant = client.beta.assistants.create(
  instructions="You are a customer support chatbot. Use your knowledge base to best respond to customer queries.",
  model="gpt-4-1106-preview",
  tools=[{"type": "retrieval"}],
  file_ids=[file.id]
)

# 쓰레드 생성 및 메세지 전달
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = "다음 문서를 확인하고, 증권사, 일반 청약자 배정물량, 최고 청약한도, 청약 증거금율을 알려주세요."
)
run = client.beta.threads.runs.create(
    thread_id = thread.id,
    assistant_id= assistant.id
)

# GPT 응답 대기 및 출력
import time

while True:
# Retrieve the run status
    run_status = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
    time.sleep(10)
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        break
    else:
        ### sleep again
        time.sleep(2)

for message in reversed(messages.data):
  print(message.role + ":" + message.content[0].text.value)

# 비용 이슈가 있기 때문에 사용후 반드시 파일 제거
file_deletion_status = client.beta.assistants.files.delete(
  assistant_id=assistant.id,
  file_id=file.id
)
