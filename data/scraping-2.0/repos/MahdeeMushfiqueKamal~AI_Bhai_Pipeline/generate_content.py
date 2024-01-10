from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sys

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") 


llm = OpenAI(temperature=0.7, max_tokens=500, openai_api_key= openai_api_key)

title = sys.argv[1]

prompt_template = PromptTemplate(
    input_variables =["title", "number_of_facts", "max_characters"],
    template="""
Imagine you making a video that describes features of a technology. 
Write {number_of_facts} interesting facts about {title}. 
The fact doesn't need to be a full sentence. 
Each fact should be on a new line. Don't use numbers or bullet points.
Each fact can have at max {max_characters} characters
""",
)

chain = LLMChain(llm = llm, prompt = prompt_template)


full_output = chain.run({"title" : title, "number_of_facts" : 10, "max_characters" : 45})

f = open("output.txt", "w")
f.write(title + "\n")
f.write(full_output)
f.close()