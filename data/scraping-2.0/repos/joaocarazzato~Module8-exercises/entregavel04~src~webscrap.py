from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import AsyncHtmlLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough

# Load HTML
url = "https://www.deakin.edu.au/students/study-support/faculties/sebe/abe/workshop/rules-safety"
loader = AsyncHtmlLoader(url)
docs = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=['main'], unwanted_tags=['footer'])

# for x in docs_transformed:
#     print(f"name: {x}")
# print(docs_transformed)

docs_text = docs_transformed[0].page_content[2330:6090]

vectorstore = FAISS.from_texts(
    [docs_text], embedding=GPT4AllEmbeddings()
)
retriever = vectorstore.as_retriever()

print(retriever)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = Ollama(model="mistral")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

print("\nModel answer:\n")
for s in chain.stream("Who is allowed to operate a lathe? What protective gear should be used to do it?"):
    print(s, end="", flush=True)