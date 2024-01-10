#!/usr/bin/env python3
import os
import openai
openai.organization = os.getenv("OPENAI_API_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")
print('OpenAI ready')