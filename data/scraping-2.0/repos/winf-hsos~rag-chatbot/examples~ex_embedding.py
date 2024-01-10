import openai

# Add and load local chatbot module
import os
import sys
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)
import chatbot as cb


# Create the embedding function
openai.api_key = cb.OPENAI_API_KEY

docs = ["Das Food Future Lab Team aus insgesamt 13 Professuren von der Fakultät Agrarwissenschaften und Landschaftsarchitektur."]
embeds = openai.Embedding.create(input=docs, model="text-embedding-ada-002")

print(embeds)