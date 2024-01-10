from openai import OpenAI
import shelve
from gradio_client import Client
from dotenv import load_dotenv
import os


load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(api_key=OPEN_AI_API_KEY)


def retrieve_chat(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        thread_id = threads_shelf.get(wa_id, None)

    if thread_id:
        # thread = client.beta.threads.retrieve(thread_id)
        messages = client.beta.threads.messages.list(thread_id=thread_id, limit=100)
        print(messages.data)
        chat = []
        for message_ in messages:
            message = {}
            message["role"] = message_.role
            message["content"] = message_.content[0].text.value
            chat.append(message)

        return chat


def evaluate_student(user_id):
    """Evaluates the student based on the Conversation between him and the chatbot"""
    conversation = retrieve_chat(user_id)
    full_chat = ""
    if conversation:
        for chat in conversation:
            full_chat += f'{chat["role"]}: {chat["content"]}'

        prompt = f"""Evaluate the student's performance in solving the coding task based on the conversation below. Offer insights into their level, identify difficulties, and provide specific recommendations for improvement:
            @conversation
            {full_chat}
            Evaluate the student's understanding of Python basics and command line arguments. Highlight specific areas of struggle and offer succinct suggestions for improvement.
        """
        client = Client("https://ise-uiuc-magicoder-s-ds-6-7b.hf.space/--replicas/dccwp/")
        response = client.predict(
                prompt,
                0,	    # Temperature
                2048,	# Max Tokens
                api_name="/evaluate_magicoder"
        )
        return response

# print(evaluate_student("task_123.mohamed"))