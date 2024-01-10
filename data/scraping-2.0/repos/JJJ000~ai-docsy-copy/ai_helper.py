import openai
from pptx import Presentation
import json

github_link = "https://github.com/jstockwin/py-pdf-parser"

qn1 = "provide 1-sentence description for this github repo https://github.com/jstockwin/py-pdf-parser "
qn2 = "provide categories for this github repo https://github.com/jstockwin/py-pdf-parser and return in array of string format, with double quote"
qn3 = "write me a tech doc for this github repo https://github.com/jstockwin/py-pdf-parser,including 1 intro paragraph and 2-4 H2 headers, in markdown format."

basic_format = """---
categories: {}
tags: {}
title: {}
linkTitle: {}
date: 2023-02-27
description: {}
---

{}

"""
def askGPT(text):
  openai.api_key = ""
  completion = openai.Completion.create(
    engine="text-davinci-003",
    prompt=text,
    max_tokens=2048,
    n=1,
    stop=None,
    temperature=0.5,
  )

  r = completion.choices[0].text
  print(r)
  print('\n')
  return r

def main():
  print("start")
  des = askGPT(qn1)
  categories = askGPT(qn2)
  body = askGPT(qn3)
  name = github_link.split("/")[-1]
  title = "\"" + name + " Github Repo Technical Documentation\""
  final = basic_format.format(categories.strip(), categories.strip(), title, title, "\"" + des.strip() + "\"", body.strip())
  
  print("done with asking openAI.")
  with open("/Users/mengting_li/Desktop/personal/ai-docsy/content/en/docs/Getting started/{}.md".format(name), "w") as f:
    f.write(final)

main()
