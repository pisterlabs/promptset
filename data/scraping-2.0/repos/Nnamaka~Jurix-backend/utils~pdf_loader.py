import os
from langchain.document_loaders import PyPDFLoader

cwd = os.path.dirname(os.path.realpath(__file__))
constitution_file = os.path.join(cwd,'constitution.pdf')

loaders = [
    PyPDFLoader(constitution_file)
]

docs = []

for loader in loaders:
    docs.extend(loader.load())