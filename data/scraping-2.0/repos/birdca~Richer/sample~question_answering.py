from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.globals import set_debug
from llm.prompt.template.openai.chat import llm_model

set_debug(True)

file = 'sample/files/NCCC_TWN_EC.CSV'
loader = CSVLoader(file_path=file)

# VectorstoreIndexCreator: Create a new model by parsing and validating input data from keyword arguments.
# Then create a vectorstore index from loaders by from_loaders() method.
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

query = """
        First column represent the year concatenate with month
        Third column represent the number of cases
        Fourth column represent the price
        which month of the year has the most number of cases? And which month of the year has the most price?
        """

# response = index.query(query)
# The answer is so wrong!
# print(response)

docs = loader.load()

embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("First row are the column names")
print(len(embed))
print(embed[:5])

# Create in memory vectorstore
# DocArrayInMemorySearch is a document index provided by Docarray that stores documents in memory.
# It is a great starting point for small datasets, where you may not want to launch a database server.
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

docs = db.similarity_search(query)
print(len(docs))
print(docs[0])

retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0.0, model=llm_model)

# stuff method: stuff all content of docs into the query
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
# response = llm.call_as_llm(f"{qdocs} Question: {query}")
# The answer is so wrong!
# print(response)

# RetrievalQA could does the same thing as above
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
# response = qa_stuff.run(query)

# DocArrayInMemorySearch query with custom llm
# response = index.query(query, llm=llm)
# The answer is so wrong!
# print(response)
