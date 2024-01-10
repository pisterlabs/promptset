#!/usr/bin/env python3
import json
import pprint

from openai import OpenAI

client = OpenAI()

assistantName = "hiro"
assistantID = "asst_vVjkIX5UzTyGROBG1fj6Lmkz"
threadID = "thread_ufYt3oNUn8dnqAP4lobD5zZp"

my_assistant = client.beta.assistants.retrieve(assistantID)

print(pprint.pformat(my_assistant.model_dump()))
