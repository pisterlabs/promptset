import argparse
import openai
import os


SYSTEM_PROMPT = '''
The user shall prompt with a question, and the system shall provide a multiple choice answer to the question. 
The answers shall be appropriate for a first year college course.
The correct response shall be listed first, followed by three incorrect responses.
Questions shall not be prefixed with a number or letter
Responses shall not be prefixed with a number or letter.
No double line breaks shall be used.
'''
conversation = [
	## Provide the model with a high level context.
    {"role": "system", "content": SYSTEM_PROMPT},
]

## Get initial message from user.
question = input( 'Question: ' )
while question != 'exit' and question != 'quit':
    ## Add new user message to end of conversation.
    conversation.append(
    {
        "role": "user", 
        "content": question
    })

    ## Query the API
    ## Change the model to whatever you want to use.
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        temperature=0.1, #default
        messages=conversation
    )

    ## Save response and print
    message = chat_completion.choices[0].message
    conversation.append(message);
    print(message.content)

    ## Get next input
    question = input( 'Question: ' )