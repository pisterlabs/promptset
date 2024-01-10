## QA chain example:

# We can also use LangChain to manage prompts for asking general questions from the LLMs. These models are proficient in addressing fundamental inquiries. Nevertheless, it is crucial to remain mindful of the potential issue of hallucinations, where the models may generate non-factual information. To address this concern, we will later introduce the Retrieval chain as a means to overcome this problem.


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")


prompt = PromptTemplate(
    template="Question: {question}\nAnswer:",
    input_variables=["question"]
)

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

chain.run("what is the meaning of life?")