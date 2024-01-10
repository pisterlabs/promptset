import os
import openai

openai.api_key = ""

knowledge_base = ''
with open('knowledge-base.txt', 'r', encoding="utf-8") as file:
    knowledge_base = file.read()

response = openai.chat.completions.create(
    model="gpt-4-1106-preview",
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": "Below is the complete documentation (https://docs.prismafinance.com/) for Prisma Finance (https://prismafinance.com). Answer the user sourcing the documentation and always return the source files used to answer a question. If answer can't be sourced from documentation then warn the user where you answer may be innacurate."
        },
        {
            "role": "user",
            "content": knowledge_base
        },
        {
            "role": "user",
            "content": '''
"when claiming rewards, how many weeks are they locked for"
"where can i vote on and collect bribes?"
"what is the difference between yprisma and cvxprisma?"
"how do redemptions work?"
"how do i know if i got redeemed?"
            '''
        }
    ],
)

bot_response = response.choices[0].message.content

print(bot_response)
