import openai
import time

def process_message(api_key, content, name="quality manager",
                    instructions="You are a quality manager. Classify types of faults and deduct possible causes.",
                    require_content="를 문제를 표현하는 부분만 나타내도록 한국어로 최대한 간결하게 압축해줘."):
    client = openai.OpenAI(api_key=api_key)

    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview"
    )

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=content+require_content
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as dongguk. The user has a premium account."
    )

    time.sleep(10)

    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        response = ""

        for msg in messages.data:
            role = msg.role
            if role != 'user':
                response += f"{msg.content[0].text.value}"
        return response