from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=1000,
  chunk_overlap=200,
  length_function=len
)
