from bimo_core import QAChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator


def test_question_answering():
    with open("tests/chains/test_document.txt") as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

    chain = QAChain(chain_type= "stuff")
    # query = "What did the president say about Justice Breyer"
    query = "What is the name of the main character in the story?"
    
    docs = docsearch.get_relevant_documents(query)
    result = chain.run(input_documents=docs, question=query)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Elara" in result

    print() # for cleaner output
    print("Query: %s" % query)
    print("Response: %s" % result)
