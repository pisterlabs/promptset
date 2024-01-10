from openai import OpenAI
import assistant


def call_api():
    assistant.run.do_run(assistant.config.get_assistant_by_name("Math Tutor"), "thread_9s3MhO8HhGAmGb5tlOtrdNWw")


def get_message():
    client = OpenAI()

    thread_messages = client.beta.threads.messages.list("thread_9s3MhO8HhGAmGb5tlOtrdNWw")

    for msg in thread_messages:
        print(msg.model_dump_json())


get_message()
