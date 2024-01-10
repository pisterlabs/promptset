from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import pandas as pd

# from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, SemanticSimilarityExampleSelector
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

prefix = """You are a knowledgeable customer service from Layanan Dosen.
Use the context below to answer various questions from users.
If you don't know the answer, just say I don't know. Don't make up an answer.
The answer given must always be in Indonesian language with a friendly tone.

Here are some examples of conversations between users and customer service to be your references:
"""

example_template = """
Question: {question}
Answer: {answer}
"""

suffix = """
Question: {input}
Answer: 
"""

# read the xlsx file
datapath = "/Users/mghifary/Work/Code/AI/data/ssd_sister_bkd.xlsx"
df = pd.read_excel(datapath)

# rename columns to lowercase
df.columns = df.columns.str.lower()

# convert the dataframe into a list of dictionaries
examples = df[['question', 'answer']].to_dict(orient='records')

# LLM
chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Activate Retrieval Augmented Generation (RAG)
# embeddings = GPT4AllEmbeddings()
embeddings = OpenAIEmbeddings()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, 
    embeddings,
    FAISS, 
    k=3 # k-nearest neighbors
)

example_prompt = PromptTemplate.from_template(example_template)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input"],
)

# Chain
chain = LLMChain(
    prompt=prompt,
    llm=chat_llm,
    # verbose=True,
)


# QA Prompts
query = "Gimana caranya isi laporan kinerja BKD tabulasi Pelaksanaan Penunjang?"
print(f"\n\n Query: {query} \n")
response = chain.predict(input=query)

query = "Portofolio dosen itu apa ya?"
print(f"\n\n Query: {query} \n")
response = chain.predict(input=query)