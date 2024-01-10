from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
"""
https://python.langchain.com/en/latest/modules/indexes/getting_started.html#
Question answering over documents consists of four steps:
    1.Create an index
    2.Create a Retriever from that index
    3.Create a question answering chain
    4.Ask questions!
"""

# Load Your Documents
loader = TextLoader('../test.txt')

# Create Your Index
index = VectorstoreIndexCreator().from_loaders([loader])

# Query Your Index
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
# " The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in
# private practice, a former federal public defender, and from a family of public school educators and police
# officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal
# Order of Police to former judges appointed by Democrats and Republicans."


# Alternatively, use query_with_sources to also get back the sources involved
query = "What did the president say about Ketanji Brown Jackson"
index.query_with_sources(query)
# {'question': 'What did the president say about Ketanji Brown Jackson', 'answer': " The president said that he
# nominated Circuit Court of Appeals Judge Ketanji Brown Jackson, one of the nation's top legal minds, to continue
# Justice Breyer's legacy of excellence, and that she has received a broad range of support from the Fraternal Order
# of Police to former judges appointed by Democrats and Republicans.\n", 'sources': '../state_of_the_union.txt'}
