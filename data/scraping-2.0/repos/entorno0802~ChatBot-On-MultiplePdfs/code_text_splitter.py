from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=30, chunk_overlap=0
)
metadata = {"source": "internet"}
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs)
