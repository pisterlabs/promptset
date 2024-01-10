import os
import openai
openai.organization = "org-"
openai.api_key = "sk-"

#openai.File.create(file=open("jsonline.jsonl", encoding="utf8"), purpose='answers')

#print (openai.File.list())

ask = openai.Answer.create(
    search_model="davinci", 
    model="curie", 
    question="what to offer to a user?", 
    file="file-6f7pNzAGltl5zcdatqRQa8jl", 
    examples_context="In 2017, U.S. life expectancy was 78.6 years.", 
    examples=[["What is human life expectancy in the United States?", "78 years."]], 
    max_rerank=10,
    max_tokens=5,
    stop=["\n", "<|endoftext|>"]
)

print(ask)