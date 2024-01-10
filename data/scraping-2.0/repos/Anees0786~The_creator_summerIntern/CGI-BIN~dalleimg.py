#!/usr/bin/python3

print("Content-type: text/html")
print()

import cgi
from langchain.llms import OpenAI
import os


#cgitb.enable()
form = cgi.FieldStorage()

cmd = form.getvalue("cmd")
#cmd="date"
#print(cmd)
mygptkey='OPEN AI KEY'

from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
)
#print(prompt)
chain = LLMChain(llm=llm, prompt=prompt)
#print(chain)
image_url = DallEAPIWrapper().run(chain.run(cmd))

print("<pre>")
print(image_url)
print("</pre>")
