#!/usr/local/bin/python3
import os
import openai
api_key = open("keys.txt")
openai.api_key = api_key.read().strip()

print(openai.Engine.list())






