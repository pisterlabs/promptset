from functools import reduce
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from query_data import *
from langchain.document_loaders import PyPDFLoader
import shutil
import os
import re
import sys
import tiktoken  # !pip install tiktoken

OUT_PAPER_PATH="data/out_paper"
INPUT_PAPER_PATH="data/input_paper"
llm = 'local'
def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


tokenizer = tiktoken.get_encoding('p50k_base')
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def remove_unnecessary_newlines(text):
    # 正则表达式: 查找换行符，其后紧跟小写字母或某些标点
    pattern = re.compile(r'\n(?=[a-z,\.])')
    text = pattern.sub(' ', text)
    # 替换这些换行符为空字符串
    return text

def translate_document(document):
    trans_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1500,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    chain = get_chain("trans", llm_name=llm)
    new_docs = trans_text_splitter.split_documents(document)
    output = ""
    for d in new_docs:
        # output += chain.run(d.page_content)
        output += chain.invoke({ "context": d.page_content })
    return output

def load_paper(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            new_filename = filename.replace(".pdf", ".txt")
            print(f"Begin loading {filename}")
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            document = loader.load()
            content = reduce(lambda x, y: x + y.page_content, document, "")
            replaced_content = remove_unnecessary_newlines(content)
            new_filepath = os.path.join(OUT_PAPER_PATH, new_filename)
            with open(new_filepath, "w", encoding="utf-8") as f:
                f.write(replaced_content)
            # translate document
            translated_text = translate_document(document)
            translated_filename = "zh_" + new_filename
            translated_path = os.path.join(OUT_PAPER_PATH, translated_filename)
            print(translated_text)
            with open(translated_path, "w", encoding="utf-8") as f:
                f.write(translated_text)
            print(f"{filename} load successfully")

def move_files_to_folder(source_folder, dst_folder):
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)
        dst_file = os.path.join(dst_folder, os.path.basename(file_path))
        shutil.move(file_path, dst_file)

if __name__ == "__main__":
    en_paper_path = "/media/dell/Samsung_T5/Paper/en_papers"
    out_en_paper_path = "/media/dell/Samsung_T5/Paper/out_en_papers"
    archive_en_papers_path = "/media/dell/Samsung_T5/Paper/archive_en_papers"
    make_path(INPUT_PAPER_PATH)
    make_path(OUT_PAPER_PATH)
    make_path(en_paper_path)
    make_path(out_en_paper_path)
    make_path(archive_en_papers_path)

    move_files_to_folder(en_paper_path, INPUT_PAPER_PATH)
    load_paper(INPUT_PAPER_PATH)
    move_files_to_folder(en_paper_path, archive_en_papers_path)
    move_files_to_folder(INPUT_PAPER_PATH, archive_en_papers_path)
    move_files_to_folder(OUT_PAPER_PATH, out_en_paper_path)