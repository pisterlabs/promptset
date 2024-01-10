import openai

file = openai.files.create(
    file=openai.file_from_path("file.pdf"),
    purpose="assistants",
)

assistant = openai.beta.assistants.create(
    model="gpt-3.5-turbo-1106",
    file_ids=[
        file.id,
    ],
    tools=[
        {"type": "retrieval"},
    ]
)

thread = openai.beta.threads.create()

print(f"File ID: {file.id}")
print(f"Assistant ID: {assistant.id}")
print(f"Thread ID: {thread.id}\n")

print("GPT: Hello! How can I assist you today?")

while True:
    message = input("You: ")
    print()

    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )

    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    while True:
        run = openai.beta.threads.runs.retrieve(
            run_id=run.id,
            thread_id=thread.id,
        )

        if run.status not in ["queued", "in_progress", "cancelling"]:
            break

    messages = openai.beta.threads.messages.list(
        thread_id=thread.id,
        limit=1,
    )
    print("GPT: " + messages.data[0].content[0].text.value)
