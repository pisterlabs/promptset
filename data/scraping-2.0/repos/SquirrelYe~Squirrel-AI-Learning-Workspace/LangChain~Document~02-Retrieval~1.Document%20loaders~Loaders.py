from langchain.document_loaders import TextLoader, CSVLoader, DirectoryLoader, UnstructuredHTMLLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import PythonLoader, BSHTMLLoader
from langchain.document_loaders import PyPDFLoader, MathpixPDFLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFium2Loader, PyMuPDFLoader, PyPDFDirectoryLoader, PDFPlumberLoader

from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import json
from pathlib import Path
from pprint import pprint

baseDir = "./files/"

# TextLoader
# 最简单的加载器将文件读入为文本，并将其全部放入一个文档中。
def TextLoaderDemo():
    loader = TextLoader(file_path=baseDir + "index.md")
    document = loader.load()
    print(document)


# CSVLoader
# CSV加载器将CSV文件读入为文档。它将每一行作为一个示例，并将每一列作为一个变量。
def CSVLoaderDemo():
    loader = CSVLoader(file_path=baseDir + "index.csv")
    document = loader.load()
    print(document)

    # 你可以使用CSVLoader的一些参数来控制如何读取CSV文件。
    # 例如，您可以指定要使用的列，或者您可以指定要跳过的行。
    loader = CSVLoader(
        file_path=baseDir + "index.csv",
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ["name", "age", "city"]
        }
    )
    document = loader.load()
    print(document)
    
    # [
        # Document(page_content='name: Alice\nage: 30\ncity: New York', metadata={'source': './files/index.csv', 'row': 0}), 
        # Document(page_content='name: Bob\nage: 25\ncity: Los Angeles', metadata={'source': './files/index.csv', 'row': 1}), 
        # Document(page_content='name: Charlie\nage: 22\ncity: San Francisco', metadata={'source': './files/index.csv', 'row': 2}), 
        # Document(page_content='name: Diana\nage: 28\ncity: Chicago', metadata={'source': './files/index.csv', 'row': 3}), 
        # Document(page_content='name: Eva\nage: 35\ncity: Boston', metadata={'source': './files/index.csv', 'row': 4})
    # ]
    
    
# DirectoryLoader
# 目录加载器将目录中的所有文件读入为文档。它将每个文件作为一个示例，并将文件名作为变量。
def DirectoryLoaderDemo():
    loader = DirectoryLoader(baseDir, glob="**/*", show_progress=True, use_multithreading=True)
    docs = loader.load()
    print(len(docs))
    
    loader = DirectoryLoader(baseDir, glob="**/*.html", show_progress=True, use_multithreading=True, loader_cls=TextLoader, silent_errors=True)
    docs = loader.load()
    print(docs)
    
    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(baseDir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()


# UnstructuredHTMLLoader
# 该加载器将HTML文件读入为文档。它将每个文件作为一个示例，并将文件名作为变量。
def UnstructuredHTMLLoaderDemo():
    loader = UnstructuredHTMLLoader(file_path=baseDir + "index.html")
    document = loader.load()
    print(document)
    
    # 我们还可以BeautifulSoup4使用BSHTMLLoader. 这将从 HTML 中提取文本到page_content，并将页面标题title提取到metadata。
    loader = BSHTMLLoader(file_path=baseDir + "index.html")
    document = loader.load()
    print(document)


# JSONLoader
# 该加载器将JSON文件读入为文档。它将每个文件作为一个示例，并将文件名作为变量。
def JSONLoaderDemo():
    data = json.loads(Path(baseDir + "index.json").read_text())
    # pprint(data)
    
    # 从数据中提取元数据
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["server_name"] = record.get("server_name")
        metadata["status"] = record.get("status")
        metadata["resource_path"] = record.get("resource_path")
        return metadata
    
    # 使用JSONLoader，我们可以将JSON文件读入为文档。
    # .data.rows[].server_name 是一个jq表达式，它将从JSON中提取服务器Name名称。
    loader = JSONLoader(
        file_path=baseDir + "index.json", 
        jq_schema='.data.rows[]',
        content_key="server_name",
        metadata_func=metadata_func
    )
    document = loader.load()
    pprint(document)


# UnstructuredMarkdownLoader
# 该加载器将Markdown文件读入为文档。它将每个文件作为一个示例，并将文件名作为变量。
def UnstructuredMarkdownLoaderDemo():
    loader = UnstructuredMarkdownLoader(file_path=baseDir + "index.md")
    document = loader.load()
    print(document)

    # 非结构化为不同的文本块创建不同的“元素”。默认情况下，我们将它们组合在一起，但您可以通过指定 轻松保持这种分离mode="elements"。
    loader = UnstructuredMarkdownLoader(file_path=baseDir + "index.md", mode="elements")
    document = loader.load()
    print(document)


# Python Loaders
# 该加载器将Python文件读入为文档。它将每个文件作为一个示例，并将文件名作为变量。
def PythonLoaderDemo():
    # 1. PyPDFLoader 这种方法的优点是可以使用页码检索文档。
    def PyPDFLoaderDemo():
        loader = PyPDFLoader(baseDir + "index.pdf")
        pages = loader.load_and_split()
        print(len(pages))
        # print(pages[0])
        chroma_index = Chroma.from_documents(pages, OpenAIEmbeddings())
        docs = chroma_index.similarity_search("介绍一下LLMSingleActionAgent?", k=2)
        for doc in docs:
            print(str(doc.metadata["page"]) + ":", doc.page_content[:], '\n')
    
    # 2. MathpixPDFLoader
    def MathpixPDFLoaderDemo():
        loader = MathpixPDFLoader(baseDir + "index.pdf")
        pages = loader.load()
        print(len(pages))

    # 3. UnstructuredPDFLoader
    def UnstructuredPDFLoaderDemo():
        loader = UnstructuredPDFLoader(baseDir + "index.pdf", mode="elements")
        pages = loader.load()
        print(pages)
    
    # 4. OnlinePDFLoader
    def OnlinePDFLoaderDemo():
        loader = OnlinePDFLoader("https://arxiv.org/pdf/2106.04473.pdf")
        pages = loader.load()
        print(pages)
        
    # 5. PyPDFium2Loader
    def PyPDFium2LoaderDemo():
        loader = PyPDFium2Loader(baseDir + "index.pdf")
        pages = loader.load()
        print(pages)
        
    # 6. PyPDFium2Loader
    def PyPDFium2LoaderDemo():
        loader = PyPDFium2Loader(baseDir + "index.pdf")
        pages = loader.load()
        print(pages)
        
    # 7. PyMuPDFLoader
    def PyMuPDFLoaderDemo():
        loader = PyMuPDFLoader(baseDir + "index.pdf")
        pages = loader.load()
        print(pages)
        
    # 8. PyPDFDirectoryLoader
    # 从目录中加载PDF文件
    def PyPDFDirectoryLoaderDemo():
        loader = PyPDFDirectoryLoader(baseDir, glob="**/*.pdf")
        pages = loader.load()
        print(pages)

    # 9. PDFPlumberLoader
    # 与 PyMuPDF 一样，输出文档包含有关 PDF 及其页面的详细元数据，并每页返回一个文档。
    def PDFPlumberLoaderDemo():
        loader = PDFPlumberLoader(baseDir + "index.pdf")
        pages = loader.load()
        print(pages)


    # 执行模块
    # PyPDFLoaderDemo()
    # MathpixPDFLoaderDemo()
    # UnstructuredPDFLoaderDemo()
    # OnlinePDFLoaderDemo()
    # PyPDFium2LoaderDemo()
    # PyPDFium2LoaderDemo()
    # PyMuPDFLoaderDemo()
    # PyPDFDirectoryLoaderDemo()
    PDFPlumberLoaderDemo()




if __name__ == "__main__":
    # TextLoaderDemo()
    # CSVLoaderDemo()
    # DirectoryLoaderDemo()
    # UnstructuredHTMLLoaderDemo()
    # JSONLoaderDemo()
    # UnstructuredMarkdownLoaderDemo()
    PythonLoaderDemo()