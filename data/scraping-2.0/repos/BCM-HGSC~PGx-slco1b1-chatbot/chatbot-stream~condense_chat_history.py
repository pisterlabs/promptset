# -*- coding:utf-8 -*-
# Created by liwenw at 9/14/23
import os
import openai
from typing import Any, Dict, List, Optional, Union
import json

def condense_history(history: List[Dict[str, str]], question: str, model: str) -> str:
    # prompt = "Given the chat history and a follow up question, rephrase the follow up question to be a standalone question, in its original language."
    prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""

    # Initialize the OpenAI API client
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    chat_history_str = json.dumps(history)
    messages = []
    messages.append({"role": "system", "content": prompt.format(chat_history=chat_history_str, question=question)})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Extract and print the assistant's response
    standalone_question = response['choices'][0]['message']['content']
    return standalone_question

def condense_history_2(history: List[Dict[str, str]], question: str, model: str) -> str:
    prompt = "Given the chat history and a follow up question, rephrase the follow up question to be a standalone question, in its original language."

    # Initialize the OpenAI API client
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    messages = []
    messages.append({"role": "system", "content": prompt})
    messages = history
    messages.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Extract and print the assistant's response
    standalone_question = response['choices'][0]['message']['content']
    return standalone_question


def main():
    chat_history =[
          {
            'role': 'user',
            'content': 'My patient takes fluvastatin for managing his cholesterol. Will a SLCO1B1 increased function affect his medication in any way?'
          },
          {
            'role': 'assistant',
            'content': "I don't know."
          },
          {
            'role': 'user',
            'content': 'This patient is also a CYP2C9 poor metabolizer'
          },
          {
            'role': 'assistant',
            'content': "Yes, an SLCO1B1 increased function can impact the effectiveness of fluvastatin, but it can potentially lead to higher than normal statin levels in blood, placing a patient at a higher risk for side effects such as muscle pain.\n\nAdditionally, being a CYP2C9 poor metabolizer means that the patient metabolizes drugs more slowly than normal. This can cause higher than normal amounts of fluvastatin in the body and thus could lead to an increased risk of side effects.\n\nTherefore, the doctor may consider adjusting the dosage or choose an alternative drug depending on the patient's condition. However, this needs to be decided by a healthcare provider. \n\nPlease note this should not be used as medical advice but only as a source of information, contact a healthcare professional for advice."
          },
          {
            'role': 'user',
            'content': 'But  does the CYP2C9 phenotype also impact fluvastatin?'
          },
          {
            'role': 'assistant',
            'content': "No, the metabolism of fluvastatin is not significantly affected by the CYP2C9 enzyme. Fluvastatin is primarily metabolized by CYP2C9, but changes in this enzyme do not significantly impact the drug's overall elimination. In contrast, many other statins are more significantly impacted by CYP2C9 variants. Therefore, a CYP2C9 poor metabolizer status should not significantly impact fluvastatin therapy. Always consult with a healthcare provider for specific medication guidance."
          },
          {
            'role': 'user',
            'content': 'What does SLCOB1 increased function mean for my patient’s fluvastatin dosage?'
          }
        ]

    question = "What does SLCOB1 increased function mean for my patient’s fluvastatin dosage?"
    standalone_question = condense_history(chat_history, question)
    print(standalone_question)

if __name__ == "__main__":
    main()