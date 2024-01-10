from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain

loader = UnstructuredFileLoader('EuronetDA.txt')
document = loader.load()

from langchain import OpenAI
llm = OpenAI(openai_api_key = "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")

#model = load_summarize_chain(llm = llm, chain_type = "stuff")
#model.run(document)

from langchain.text_splitter import RecursiveCharacterTextSplitter
char_text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
docs = char_text_splitter.split_documents(document)

#model = load_summarize_chain(llm = llm, chain_type = "map_reduce")
#model.run(docs)

model = load_summarize_chain(llm = llm, chain_type = "refine")
output = model.run(docs)
print(output)