# from openai import OpenAI
# import os
# import httpx
# HTTP_PROXY = "http://127.0.0.1:7890"
# client = OpenAI(
#     http_client=httpx.Client(
#         proxies=HTTP_PROXY,
#         transport=httpx.HTTPTransport(local_address="0.0.0.0"),
#     ),
#     api_key=os.environ.get("OPENAI_API_KEY"))
# # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
# # if you saved the key under a different environment variable name, you can do something like:
# # client = OpenAI(
# #   api_key=os.environ.get("CUSTOM_ENV_NAME"),
# # )
#
# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "user",
#     "content": "Say this is a test",}
#   ]
# )
#
# print(completion.choices[0].message)