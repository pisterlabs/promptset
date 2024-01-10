import base64
import io
import mimetypes
import os
import tempfile
import pandas as pd
import requests
from openai import OpenAI
from pathlib import Path

from engine.models import Chatbots

client = OpenAI()


def create_assistant(file, chatbot_id):
    """
    You currently cannot set the temperature for Assistant via the API.
    """
    instance = Chatbots.objects.get(chatbot_id=chatbot_id)

    assistant = client.beta.assistants.create(
        name=instance.title,
        instructions=instance.instructions,
        tools=[{"type": "retrieval"}],
        model="gpt-4-1106-preview",
        file_ids=[file.id],
    )
    instance.assistant_id = assistant.id
    instance.save()
