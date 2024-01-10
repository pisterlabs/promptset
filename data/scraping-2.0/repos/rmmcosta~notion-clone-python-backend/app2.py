# imports
import time  # for measuring time duration of API calls
from openai import OpenAI
client = OpenAI()  # for OpenAI API calls
prompt = "cats"

# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/guides/chat

# record the time before the request is sent
start_time = time.time()

# send a ChatCompletion request to count to 100
completion = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages = [
        {
            'role': 'system',
            'content': '''You are a helpful AI embedded in a notion text editor app that is used to autocomplete sentences.
            The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
            AI is a well-behaved and well-mannered individual.
            AI is always friendly, kind, and inspiring, and he is eager to provide vivid and thoughtful responses to the user.'''
        },
        {
            'role': 'user',
            'content': '''I am writing a piece of text in a notion text editor app.
            Help me complete my train of thought here: ##{}##
            keep the tone of the text consistent with the rest of the text.
            keep the response short and sweet.'''.format(prompt)
        },
    ],
    stream=True  # again, we set stream=True
)
# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
# iterate through the stream of events
for chunk in completion:
    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
    collected_chunks.append(chunk)  # save the event response
    chunk_message = chunk.choices[0].delta.content  # extract the message
    collected_messages.append(chunk_message)  # save the message
    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

# print the time delay and text received
print(f"Full response received {chunk_time:.2f} seconds after request")
# clean None in collected_messages
collected_messages = [m for m in collected_messages if m is not None]
full_reply_content = ''.join([m for m in collected_messages])
print(f"Full conversation received: {full_reply_content}")