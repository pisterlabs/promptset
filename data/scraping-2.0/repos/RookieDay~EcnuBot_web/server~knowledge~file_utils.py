import pandas as pd
import os

def sheet_to_string(sheet, sheet_name=None):
    result = []
    for index, row in sheet.iterrows():
        row_string = ""
        for column in sheet.columns:
            row_string += f"{column}: {row[column]}, "
        row_string = row_string.rstrip(", ")
        row_string += "."
        result.append(row_string)
    return result


def excel_to_string(file_path):
    # 读取Excel文件中的所有工作表
    excel_file = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)

    # 初始化结果字符串
    result = []

    # 遍历每一个工作表
    for sheet_name, sheet_data in excel_file.items():
        # 处理当前工作表并添加到结果字符串
        result += sheet_to_string(sheet_data, sheet_name=sheet_name)

    return result


def file_loader(file_path, file_type):
    from langchain.schema import Document
    from langchain.text_splitter import TokenTextSplitter

    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=30)
    documents = []
    try:
        if file_type == ".pdf":
            from langchain.document_loaders import PyPDFLoader

            loader = PyPDFLoader(file_path)
            file_loader = loader.load()
        elif file_type == ".docx":
            print("Loading Word...")
            from langchain.document_loaders import UnstructuredWordDocumentLoader

            loader = UnstructuredWordDocumentLoader(file_path)
            file_loader = loader.load()
        elif file_type == ".pptx":
            print("Loading PowerPoint...")
            from langchain.document_loaders import UnstructuredPowerPointLoader

            loader = UnstructuredPowerPointLoader(file_path)
            file_loader = loader.load()
        elif file_type == ".epub":
            print("Loading EPUB...")
            from langchain.document_loaders import UnstructuredEPubLoader

            loader = UnstructuredEPubLoader(file_path)
            file_loader = loader.load()
        elif file_type == ".xlsx":
            print("Loading Excel...")
            text_list = excel_to_string(file_path)
            file_loader = []
            for elem in text_list:
                file_loader.append(
                    Document(page_content=elem, metadata={"source": file_path})
                )
        else:
            print("Loading text file...")
            from langchain.document_loaders import TextLoader

            loader = TextLoader(file_path, "utf8")
            file_loader = loader.load()
    except Exception as e:
        file_loader = "Loading failed"
    print(f"docs: {file_loader}")
    if file_loader is not None:
        texts = text_splitter.split_documents(file_loader)
        documents.extend(texts)
    return documents

def gradioFile_loader(file_dict, file_Basepath):
    from langchain.schema import Document
    from langchain.text_splitter import TokenTextSplitter

    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=30)
    documents = []
    for file_key in file_dict:
        file_path = os.path.join(file_Basepath, f"{str(file_key)}")
        print('utils....')
        file_type = file_dict[file_key]['file_type']
        print(file_path)
        print(file_type)
        try:
            if file_type == ".pdf":
                from langchain.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                file_loader = loader.load()
            elif file_type == ".docx":
                print("Loading Word...")
                from langchain.document_loaders import UnstructuredWordDocumentLoader

                loader = UnstructuredWordDocumentLoader(file_path)
                file_loader = loader.load()
            elif file_type == ".pptx":
                print("Loading PowerPoint...")
                from langchain.document_loaders import UnstructuredPowerPointLoader

                loader = UnstructuredPowerPointLoader(file_path)
                file_loader = loader.load()
            elif file_type == ".epub":
                print("Loading EPUB...")
                from langchain.document_loaders import UnstructuredEPubLoader

                loader = UnstructuredEPubLoader(file_path)
                file_loader = loader.load()
            elif file_type == ".xlsx":
                print("Loading Excel...")
                text_list = excel_to_string(file_path)
                file_loader = []
                for elem in text_list:
                    file_loader.append(
                        Document(page_content=elem, metadata={"source": file_path})
                    )
            else:
                print("Loading text file...")
                from langchain.document_loaders import TextLoader

                loader = TextLoader(file_path, "utf8")
                file_loader = loader.load()
        except Exception as e:
            file_loader = "Loading failed"
        print(f"docs: {file_loader}")
        if file_loader is not None:
            texts = text_splitter.split_documents(file_loader)
            documents.extend(texts)
    return documents
