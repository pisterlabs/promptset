# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module is for running the travel agent use case."""

from langchain.memory import ChatMessageHistory
import models, prompts

import warnings
warnings.filterwarnings("ignore")


def run_travel_agent(text: str):
    messages = prompts.setup_travel_agent_messages(text)
    history.add_message(messages[0])
    history.add_message(messages[1])
    chat_model = models.set_openai_chat_model()
    response = chat_model(history.messages)
    return response


if __name__ == "__main__":
    history = ChatMessageHistory()
    history.add_ai_message("Hello, I'm your travel agent. What can I do for you?")
    text = "I want to go to the Allgau area in Germany. Can you build me an itinerary for 8 days?"
    ai_response = run_travel_agent(text)
    history.add_ai_message(ai_response.content)

    lines = ai_response.content.splitlines()
    for line in lines:
        print(line)

    print('\n\n', history.messages)




