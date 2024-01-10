from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import PyPDFLoader
import os
from docx import Document
import subprocess
import pickle
import tiktoken  # !pip install tiktoken
import sys
import re
import os



def convert_docx_to_pdf_and_delete(source_folder):
    # List all files in the directory
    files = os.listdir(source_folder)

    # Filter out only .doc and .docx files
    doc_files = [f for f in files if f.endswith(('.docx', '.doc'))]

    for doc_file in doc_files:
        # Full path to the doc/docx file
        input_file_path = os.path.join(source_folder, doc_file)
        output_file_path = os.path.join(source_folder, os.path.splitext(doc_file)[0] + '.pdf')
        # Convert using libreoffice
        subprocess.call(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', source_folder, input_file_path])

        # Delete the .doc or .docx file
        os.remove(input_file_path)

# Provide the path to your folder


def load_pdf(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"Begin loading {filename}")
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            document = loader.load()
            documents.extend(document)
            print(f"{filename} load successfully")
    return documents


def delete_space(path):
    import os

    # 指定文件夹路径
    folder_path = path

    # 遍历文件夹中的所有文件
# 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 只处理.txt文件
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # 打开文件并读取内容
                with open(file_path, 'r') as f:
                    content = f.read()

                # 去掉所有的空格和换行符
                processed_content = content.replace(' ', '').replace('\n', '')

                # 重新写入文件
                with open(file_path, 'w') as f:
                    f.write(processed_content)

    print("处理完成!")



def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('p50k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def clean_string(input_doc):
    placeholder = "<DOUBLE_NEWLINE>"
    step1 = input_doc.page_content.replace("\n\n", placeholder)
    step2 = step1.replace("\n", "").replace(" ", "").replace(".", "").replace('\x00', "")
    result = step2.replace(placeholder, "\n\n")
    cleaned_text = re.sub('[\ue000-\uf8ff]', '', result)
    final_text = re.sub(r'\s+', "", cleaned_text)
    threshold = 10
    cleaned_content = re.sub(
        r'[A-Za-z0-9$%!@#^&*]{' + str(threshold) + ',}', '', final_text)
    input_doc.page_content = cleaned_content
    input_doc.metadata = input_doc.metadata
    return input_doc



def main():

    PATH = "data/vector_src"
    print("Loading data...")
    # loader = UnstructuredFileLoader("state_of_the_union.txt")
    if len(sys.argv) < 2:
        print("Plese input the dir to be ingested")
        sys.exit(0)

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    DIR_SAVED = PATH + "/" + sys.argv[1] + "-vectorstore.pkl"
    if os.path.exists(DIR_SAVED):
        return
    folder_path = "data/docs/" + sys.argv[1]  # path to xx_src
    convert_docx_to_pdf_and_delete(folder_path)
    loader = PyPDFDirectoryLoader(folder_path)
    text_docs = DirectoryLoader(path=folder_path, glob="**/*.txt").load()
    # delete_space(folder_path)
    # raw_documents = load_pdf(sys.argv[1])
    raw_documents = loader.load()
    raw_documents.extend(text_docs)

    tokenizer = tiktoken.get_encoding('p50k_base')
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=50,
        length_function=tiktoken_len,
    )
    documents = text_splitter.split_documents(raw_documents)

    documents = list(map(lambda doc: clean_string(doc), documents))
    print(documents[0])
    print("Creating vectorstore...")
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="../models/stella-large-zh-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    with open(DIR_SAVED, "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    main()
