#!/usr/bin/env python3

#from rich.text import Text
import time

import bconfig
import bmodel
import bui

def call_openai_api(prompt_messages, callback):

    message = ''

    prompt_messages = [{
        'role': message['role'],
        'content': message['content'],
    } for message in prompt_messages ]

    try:
        import openai
        from apikey import OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY

        model = bmodel.models.get(bconfig.model)
        if model is None:
            msg = f"Model {bconfig.model} not found."
            bui.print(msg, style=bui.style("error"))
            exit(1)

        # record the time before the request is sent
        start_time = time.time()

        # send a ChatCompletion request
        # https://platform.openai.com/docs/guides/chat
        max_tokens = bconfig.max_response_tokens
        response = openai.ChatCompletion.create(
            model=model["name"],
            temperature=0.8,
            max_tokens=max_tokens,
            stream=True,
            messages=prompt_messages,
        )

        # iterate through the stream of events
        for chunk in response:
            chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            content = chunk_message.get('content', '')
            message = message + content
            callback(message)

        # print the time delay and text received
        #print(f"Full response received {chunk_time:.2f} seconds after request")

    except ImportError:
        msg = "Please run ./install.sh to configure your OpenAI API key."
        bui.print(msg, style=bui.style("error"))
        exit(1)

    except Exception as e:
        result = []
        if message != '':
            result.append(message)
        result.append(str(e))
        callback('\n\n'.join(result))


