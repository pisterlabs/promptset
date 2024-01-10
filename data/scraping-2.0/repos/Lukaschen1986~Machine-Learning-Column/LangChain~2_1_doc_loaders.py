# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/data_connection/document_loaders/
https://www.langchain.com.cn/modules/indexes/document_loaders

Use document loaders to load data from a source as Document's.
"""
import os
import json
from pathlib import Path
from pprint import pprint
from bs4 import BeautifulSoup
import torch as th
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (TextLoader, DirectoryLoader, UnstructuredHTMLLoader,
                                        BSHTMLLoader, JSONLoader, UnstructuredMarkdownLoader,
                                        PyPDFLoader, MathpixPDFLoader, UnstructuredPDFLoader,
                                        OnlinePDFLoader, PyPDFium2Loader, PDFMinerLoader,
                                        PDFMinerPDFasHTMLLoader, PyMuPDFLoader,
                                        PyPDFDirectoryLoader, PDFPlumberLoader,
                                        AmazonTextractPDFLoader, DataFrameLoader)


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# Get started
loader = TextLoader(file_path=os.path.join(path_data, "README.md"))
loader.load()

# ----------------------------------------------------------------------------------------------------------------
# csv
loader = CSVLoader(file_path=os.path.join(path_data, "test_csv.csv"))
data = loader.load()

loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', 
                   csv_args={'delimiter': ',',
                             'quotechar': '"',
                             'fieldnames': ['MLB Team', 'Payroll in millions', 'Wins']})
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', 
                   source_column="Team")
data = loader.load()

# ----------------------------------------------------------------------------------------------------------------
# DataFrame
loader = DataFrameLoader(df, page_content_column="Team")
data = loader.load()

# ----------------------------------------------------------------------------------------------------------------
# File Directory
# We can use the glob parameter to control which files to load.
loader = DirectoryLoader('../', glob="**/*.md")
docs = loader.load()

# ----------------------------------------------------------------------------------------------------------------
# HTML
loader = UnstructuredHTMLLoader("example_data/fake-content.html")
data = loader.load()

loader = BSHTMLLoader("example_data/fake-content.html")
data = loader.load()

# ----------------------------------------------------------------------------------------------------------------
# JSON
file_path = './example_data/facebook_chat.json'
data = json.loads(Path(file_path).read_text())
pprint(data)

loader = JSONLoader(file_path='./example_data/facebook_chat.json',
                    jq_schema='.messages[].content',  # The jq schema to use to extract the data or text from the JSON.
                    text_content=False)  # Boolean flag to indicate whether the content is in string format, default to True.
data = loader.load()

# ----------------------------------------------------------------------------------------------------------------
# Markdown
markdown_path = "../../../../../README.md"
loader = UnstructuredMarkdownLoader(markdown_path)
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
data = loader.load()

# ----------------------------------------------------------------------------------------------------------------
# PDF
'''
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pypdf rapidocr-onnxruntime
'''
loader = PyPDFLoader(file_path="example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()

loader = PyPDFLoader(file_path="https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
pages = loader.load()
pages[4].page_content

loader = MathpixPDFLoader(file_path="example_data/layout-parser-paper.pdf")
data = loader.load()

loader = UnstructuredPDFLoader(file_path="example_data/layout-parser-paper.pdf")
loader = UnstructuredPDFLoader(file_path="example_data/layout-parser-paper.pdf", mode="elements")
data = loader.load()
data[0]

loader = OnlinePDFLoader(file_path="https://arxiv.org/pdf/2302.03803.pdf")
data = loader.load()

loader = PyPDFium2Loader(file_path="example_data/layout-parser-paper.pdf")
data = loader.load()

loader = PDFMinerLoader(file_path="example_data/layout-parser-paper.pdf")
data = loader.load()

loader = PDFMinerPDFasHTMLLoader("example_data/layout-parser-paper.pdf")
data = loader.load()[0]   # entire PDF is loaded as a single Document
soup = BeautifulSoup(data.page_content,'html.parser')
content = soup.find_all('div')

loader = PyMuPDFLoader("example_data/layout-parser-paper.pdf")
data = loader.load()

loader = PyPDFDirectoryLoader("example_data/")
docs = loader.load()

loader = PDFPlumberLoader("example_data/layout-parser-paper.pdf")
data = loader.load()

loader = AmazonTextractPDFLoader("example_data/alejandro_rosalez_sample-small.jpeg")
documents = loader.load()



