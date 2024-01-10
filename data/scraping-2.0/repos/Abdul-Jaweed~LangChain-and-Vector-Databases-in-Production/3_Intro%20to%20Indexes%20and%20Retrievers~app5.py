from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("The One Page Linux Manual.pdf")
pages = loader.load()


from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

texts = text_splitter.split_documents(pages)

print(texts[0])

print(f"You have {len(texts)} documents")
print("Preview : ")
print(texts[0].page_content)

