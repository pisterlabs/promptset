import openai
from secrets import SECRET_API_KEY
openai.api_key = SECRET_API_KEY   


# Get the GPT response to the user's input
# gpt_response = openai.ChatCompletion.create(
#     model="gpt-4",
#     messages=[
#         {
#             "role": "system",
#             "content": system_description
#         },
#         *chat_log
#     ]
# )