#!/usr/bin/env python3
import openai
import sys
import os
from pyln.client import Plugin
# from lightning import Plugin
plugin = Plugin()

# plugin.add_option("advice", "Read logs and suggest other advices for working with lightning-cli")
# plugin.add_option("help", "Tell you appropriate commands for your query")

class LightningError(Exception):
  """Exception raised when there is an error in the lightningGPT plugin."""
  plugin.log("LightningGPT error: {}".format(sys.exc_info()[1]))

@plugin.init()
def init(plugin, options, **kwargs):
  if not os.environ['OPENAI_API_KEY']:
    raise LightningError("Set your OPENAI_API_KEY in the Secrets in Tools Section of your replit")
    sys.exit(1)
  else:
    plugin.log("OPENAI_API_KEY detected as environment variable 🎉 ")
  plugin.log("LightningGPT plugin intialized...")


@plugin.method("helpGPT")
def helpGPT(plugin,command=None, *args):
  """Gives you appropriate commands or help for your query"""
  
  text=""
  file_paths=[f"{os.path.realpath(os.path.dirname(__file__))}"+"/cheatsheet.md"]
  for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
      text += file.read()
    
  query = f"""You are a ligtningNodeGPT, friendly and helpful AI assistant by Amit0617 that provides help with operating lightning nodes for btc. You give thorough answers with command examples if possible.

QUESTION: How to merge tables in pandas?
=========
Content: pandas provides various facilities for easily combining together Series or DataFrame with various kinds of set logic for the indexes and relational algebra functionality in the case of join / merge-type operations.
Source: 28-pl
Content: pandas provides a single function, merge(), as the entry point for all standard database join operations between DataFrame or named Series objects: \n\npandas.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
Source: 30-pl
=========
FINAL ANSWER: To merge two tables in pandas, you can use the pd.merge() function. The basic syntax is: \n\npd.merge(left, right, on, how) \n\nwhere left and right are the two tables to merge, on is the column to merge on, and how is the type of merge to perform. \n\nFor example, to merge the two tables df1 and df2 on the column 'id', you can use: \n\npd.merge(df1, df2, on='id', how='inner')
SOURCES: 28-pl 30-pl

QUESTION: How are you?
=========
CONTENT:
SOURCE:
=========
FINAL ANSWER: I am fine, thank you. How are you?
SOURCES:

Question: {command}
=========
{text}
=========
FINAL ANSWER:

"""
  response = openai.ChatCompletion.create(
      messages=[
          {'role': 'system', 'content': 'You answer questions about the lightning'},
          {'role': 'user', 'content': query},
      ],
      model='gpt-3.5-turbo',
      temperature=0,
  )
  
  print(response['choices'][0]['message']['content'])

# if __name__ == '__main__':
plugin.run()