# Text Summarization

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0
)


# Next, we define a prompt template for summarization:

summarization_template = "Summarize the following text to one sentence: {text}"
summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template=summarization_template
)
summarization_chain = LLMChain(
    llm=llm,
    prompt=summarize_prompt
)

# To use the summarization chain, simply call the predict method with the text to be summarized:

text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."

# To use the summarization chain, simply call the predict method with the text to be summarized:

summarize_text = summarization_chain.predict(text=text)

