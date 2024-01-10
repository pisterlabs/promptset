import sys
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains.summarize import load_summarize_chain

# Load the document
loader = UnstructuredPDFLoader(str(sys.argv[1]))
data = loader.load()

llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="stuff")  # "refine" or "map_reduce"

result = chain.run(data)

print(result)
