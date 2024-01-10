from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import pprint
import google.generativeai as palm

repo_path = "/Users/akhilkumarp/development/personal/github/CodeWithAI-Python"
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*.py",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
palm.configure(api_key='AIzaSyD3d3npOFWRAbYPaKP1Yk8KWhGyiHtumZM')
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
# prompt = """
# You are an expert at solving word problems.

# Solve the following problem:

# I have three houses, each with three cats.
# each cat owns 4 mittens, and a hat. Each mitten was
# knit from 7m of yarn, each hat from 4m.
# How much yarn was needed to make all the items?

# Think about it step by step, and show your work.
# """

# completion = palm.generate_text(
#     model='models/text-bison-001',
#     prompt=prompt,
#     temperature=0,
#     # The maximum length of the response
#     max_output_tokens=800,
# )
import requests
import os

def call_palm_api(prompt):
    body = {
        "prompt": {"text": prompt},
    }
    url = f'https://generativelanguage.googleapis.com/v1beta3/models/text-bison-001:generateText?key=AIzaSyDYTaObtIda0gQKytXUpWlDU0AsPSR4Gvo'
    print(body)
    try:
        response = requests.post(url, json=body)
        response.raise_for_status()
        res = response.json()
        res = res['candidates'][0]['output']
        return res
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")

call_palm_api("Explain advantages of liver and generate the response in markdown format")
for text in texts:
    print(text)
    prompt = 'Explain this code '+text.page_content
    call_palm_api(prompt)
    completion = palm.generate_text(
        model='models/text-bison-001',
        prompt=prompt,
        temperature=0.5,
        # The maximum length of the response
        max_output_tokens=4000,
    )
    print(completion.result)


print("Hello")


# weaviate_url = "http://localhost:8080"
# embeddings = []  # Replace with your own embeddings
# docsearch = Weaviate.from_texts(
#     texts,
#     embeddings,
#     weaviate_url=weaviate_url,
#     by_text=False,
#     metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
# )
# retriever = docsearch.as_retriever()

# template = """
# You are an assistant for generating documentation over code. Use the following pieces of retrieved context to generate the documentation. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:
# """

# prompt = ChatPromptTemplate.from_template(template)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# generated_documentation = rag_chain.invoke("How does this function work?")

