from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

print([e.value for e in Language])
"""
['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 'python', 'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex', 'html', 'sol', 'csharp', 'cobol']
"""

separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
print(separators)
"""
['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
"""

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs)
"""
[Document(page_content='def hello_world():\n    print("Hello, World!")'), Document(page_content='# Call the function\nhello_world()')]
"""