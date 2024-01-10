import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

# Similarity search

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

texts = [
    """Le capital est octroyé une fois par année pour une hospitalisation de plus de 24 heures..""",
    """Un capital bonus est alloué une fois par année.""",
    """Souscription au capital possible jusqu’à 50 ans.""",
]
print(f'Texts: {texts}')
smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Quand est-ce que le capital est octroyé ?"

print(f'Question: "{question}"')

# print similarity search results
print("Similarity search results:")
print(smalldb.similarity_search(question, k=2))
print()

# Addressing Diversity: Maximum marginal relevance
print("Max marginal relevance search results:")
print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))


# Addressing Specificity: working with metadata using self-query retriever

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/Assura-Basis_CGA_LAMal_2024_F.pdf`, `docs/NEW_ASSURA_CSC-Natura_2015.07_F.pdf`, or `docs/Vue_Ensemble_Produits_F_V33_08.2023.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "Qu'en est-il de la souscription ?"

docs = retriever.get_relevant_documents(question)

for d in docs:
    print(d['metadata'])


# Additional tricks: compression

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# Wrap our vectorstore
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr") # use mmr to avoid duplicates
)

question = "qu'en est-il à propos des frais administratifs?"
print(f'Question: "{question}"')
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)


# Other types of retrieval

from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("docs/Vue_Ensemble_Produits_F_V33_08.2023.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
splits = text_splitter.split_text(joined_page_text)

# Retrieve
svm_retriever = SVMRetriever.from_texts(splits,embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

question = "qu'en est-il à propos des frais administratifs?"
docs_svm=svm_retriever.get_relevant_documents(question)
print("SWMRetriver results:")
print(docs_svm[0])

print("- - -")
question = "Qu'en est-il de la souscription ?"
docs_tfidf=tfidf_retriever.get_relevant_documents(question)
print("TFIDFRetriever results:")
print(docs_tfidf[0])
