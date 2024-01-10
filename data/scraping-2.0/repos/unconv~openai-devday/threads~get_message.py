import openai

messages = openai.beta.threads.messages.list(
    thread_id="",
    limit=1,
)

print(messages.data[0].content[0].text.value)