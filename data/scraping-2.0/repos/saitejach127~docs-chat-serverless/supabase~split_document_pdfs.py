from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter

DOCUMENT_PATH = "documents/csm.pdf"

print("start pdf load")
loader = PyPDFLoader(DOCUMENT_PATH)
print("loaded pdf")
pages = loader.load_and_split()

print("split to pages")
text_splitter = NLTKTextSplitter(chunk_size=1000)

print("splitter done")
batch = [x.page_content for x in pages]
batches = []
for t in batch:
    batches.extend(text_splitter.split_text(t))
batch=batches
print("for loop done adding to file")

with open("splittedtext.txt", 'w') as file:
    file.write("$$$".join(batches))

print("added to file")