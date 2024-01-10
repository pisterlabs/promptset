# -*- coding:utf-8 -*-
# Created by liwenw at 9/13/23

import os
import openai
from typing import Any, Dict, List, Optional, Union


def condense_questions(questions: List[str], question: str, model: str) -> str:
    instruction = "Combine the user's questions in chat history and current question into a standalone question."

    # Initialize the OpenAI API client
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    messages = [{"role": "system", "content": instruction},]
    for message in questions:
        messages.append({"role": "user", "content": message})
    messages.append({"role": "user", "content": question})

    # Send the condensed conversation to the API
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Extract and print the assistant's response
    standalone_question = response['choices'][0]['message']['content']
    return standalone_question

def main():
    questions = [
        "My patient takes fluvastatin for managing his cholesterol. Will a SLCO1B1 increased function affect his medication in any way?",
        "This patient is also a CYP2C9 poor metabolizer",
        "But  does the CYP2C9 phenotype also impact fluvastatin?",
    ]
    new_question = "What does SLCOB1 increased function mean for my patient’s fluvastatin dosage?"

    standalone = condense_questions(questions, new_question)
    print(standalone)

if __name__ == "__main__":
    main()
