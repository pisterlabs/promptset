
"""
Test module for chat_v2 function.
"""

import os
from openai import OpenAI
from grafana_openai_monitoring import chat_v2

def test_chat_v2():
    """
    Test the chat_v2 functionality with OpenAI API.

    This test function sets up the OpenAI API with monitoring using the chat_v2 decorator,
    sends a sample chat message, and asserts the response.

    Make sure you have the required environment variables set:
    - OPENAI_API_KEY
    - PROMETHEUS_URL
    - LOKI_URL
    - PROMETHEUS_USERNAME
    - LOKI_USERNAME
    - GRAFANA_CLOUD_ACCESS_TOKEN
    """

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Apply the custom decorator to the OpenAI API function
    client.chat.completions.create = chat_v2.monitor(
        client.chat.completions.create,
        metrics_url=os.getenv("PROMETHEUS_URL"),
        logs_url=os.getenv("LOKI_URL"),
        metrics_username=os.getenv("PROMETHEUS_USERNAME"),
        logs_username=os.getenv("LOKI_USERNAME"),
        access_token=os.getenv("GRAFANA_CLOUD_ACCESS_TOKEN")
    )

    # Now any call to openai.ChatCompletion.create will be automatically tracked
    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                            max_tokens=100,
                                            messages=
                                                [
                                                    {
                                                        "role": "user",
                                                        "content": "What is Grafana?"
                                                    }
                                                ]
                                            )

    assert response.object == 'chat.completion'
