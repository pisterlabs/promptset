#!/usr/bin/python3

import cgi

print("content-type: text/html")
print()

data=cgi.FieldStorage()
tb=data.getvalue("name")
from langchain.llms import OpenAI
myllm=OpenAI(
    model="text-davinci-003",
    openai_api_key="",
    temperature=0

)
res=myllm.generate( prompts=[tb])
print(res.generations[0][0].text)
