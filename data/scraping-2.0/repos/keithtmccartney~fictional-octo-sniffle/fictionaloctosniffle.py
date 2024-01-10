import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# --------------------------------------------

# from langchain.llms import OpenAI

# llm = OpenAI(model_name="text-davinci-003") # The meaning of life is subjective and can vary from person to person, but is generally thought of as a purpose or reason to live. It can involve finding fulfillment, developing personal connections, and achieving goals. Ultimately, it is up to each individual to discover and decide what their own meaning of life is.

# question = "What is the meaning of life?"

# print(llm(question))

# --------------------------------------------

from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain

template = "What are the top {k} resources to learn {this} in 2023?"

llm = OpenAI(model_name="text-davinci-003")

prompt = PromptTemplate(template=template, input_variables=["k", "this"])

chain = LLMChain(llm=llm, prompt=prompt)

input = {"k": 10, "this": "Python"}

print(chain.run(input))

# 1. Codecademy
# 2. Coursera
# 3. Udemy
# 4. edX
# 5. Python.org
# 6. LearnPython.org
# 7. Google’s Python Class
# 8. Python for Everybody Specialization
# 9. Automate the Boring Stuff with Python
# 10. The Python Bible by Ziyad Yehia
