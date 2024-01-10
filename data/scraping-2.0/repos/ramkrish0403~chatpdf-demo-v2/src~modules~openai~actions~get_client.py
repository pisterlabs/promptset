# from openai import OpenAI

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     # api_key="My API Key",
# )

import openai
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")
