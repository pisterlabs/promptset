# Creating a Question-Answering Prompt Template

# Let's create a simple question-answering prompt template using LangChain

import warnings
warnings.filterwarnings('ignore')

from langchain.prompts import PromptTemplate

template = """Questions: {question}
Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question

question = "What is the capital city of France?"


from langchain import HuggingFaceHub, LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

huggingface_apikey = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# initialize Hub LLM

hub_llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingface_apikey,
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain

llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# print(llm_chain.run(question))



# Asking Multiple Questions


qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]
res = llm_chain.generate(qa)
# print( res )


# We can modify our prompt template to include multiple questions to implement a second approach. The language model will understand that we have multiple questions and answer them sequentially. This method performs best on more capable models.

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""

long_prompt = PromptTemplate(
    template=multi_template,
    input_variables=['questions']
)

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=hub_llm
)

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
		"What color is a ripe banana?\n"
)

print(llm_chain.run(qs_str))

