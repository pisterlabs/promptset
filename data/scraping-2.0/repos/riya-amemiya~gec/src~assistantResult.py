from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI()
status = "queued"

while status != "completed":
    run = client.beta.threads.runs.retrieve(
        thread_id="thread_ySH3wSeXPJpbnB4EYCPpwcL3",
        run_id="run_SavL5mlQmiY0k4eF7sv3g3IB",
    )
    status = run.status

messages = client.beta.threads.messages.list(
    thread_id="thread_ySH3wSeXPJpbnB4EYCPpwcL3"
)

for message in reversed(messages.data):
    print(message.content[0].text.value)
