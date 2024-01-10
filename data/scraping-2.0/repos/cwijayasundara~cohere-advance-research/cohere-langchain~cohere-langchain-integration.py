from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
import os

os.environ["COHERE_API_KEY"] = ""

loader = TextLoader("/Users/chamindawijayasundara/Documents/self_learn/cohere-advanced-research/docs/steve-jobs"
                    "-commencement.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts[0])

embeddings = CohereEmbeddings(model="multilingual-22-12")
db = Qdrant.from_documents(texts, embeddings, location=":memory:", collection_name="my_documents", distance_func="Dot")

questions = [
           "What did the author liken The Whole Earth Catalog to?",
           "What was Reed College great at?",
           "What was the author diagnosed with?",
           "What is the key lesson from this article?",
           "What did the article say about Michael Jackson?",
           ]

prompt_template = """Text: {context}
Question: {question} Answer the question based on the text provided. If the text doesn't contain the answer, 
reply that the answer is not available."""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=Cohere(model="command-nightly", temperature=0),
                                 chain_type="stuff",
                                 retriever=db.as_retriever(),
                                 chain_type_kwargs=chain_type_kwargs,
                                 return_source_documents=True)

for question in questions:
    answer = qa({"query": question})
    print(answer)
