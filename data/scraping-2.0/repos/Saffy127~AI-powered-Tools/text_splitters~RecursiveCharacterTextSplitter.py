from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# LLMs have limited context length.
# This means that we will need to pass in split pieces of documents to our models, and to do that, we can use LangChains text splitters

loader = TextLoader("data/ai.txt")

documents = loader.load()

#Now we create our text splitter.

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 100,
  chunk_overlap = 0
)

# Now lets use the text splitter to split up documents.
texts = text_splitter.split_documents(documents)

# Now we have all of these chunks saved.
print(texts[0])
print(texts[1])
