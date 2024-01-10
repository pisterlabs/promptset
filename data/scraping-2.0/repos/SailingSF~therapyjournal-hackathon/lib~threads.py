import os
from openai import OpenAI
from dotenv import load_dotenv
from asgiref.sync import sync_to_async

from .open_ai_tools import get_open_ai_client

load_dotenv()


def create_thread(user, open_ai_client):
    """
    Creates a thread for the given user on the current OpenAI Assistant
    Used when a user does not have an active thread with an Assistant
    """

    print("in create threat")
    thread = open_ai_client.beta.threads.create()

    return thread


async def get_or_create_thread(user):
    """
    Takes a user id parameter and finds the thread on this program's OpenAI Assistant for that user
    If the user does not have a thread it calls the create thread function
    """

    open_ai_client = get_open_ai_client()
    if user.thread_id:
        # try to find thread for user, if no thread it creates one

        thread_id = user.thread_id
        thread = open_ai_client.beta.threads.retrieve(thread_id)
    else:
        print("Creating new thread", user, open_ai_client)
        thread = create_thread(user, open_ai_client)
        user.thread_id = thread.id
        await sync_to_async(user.save)()

    return thread
