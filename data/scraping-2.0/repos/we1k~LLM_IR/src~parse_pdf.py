import re
import sys
import random

from collections import defaultdict

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter, CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def contains_digit(sentence):
    for char in sentence:
        if char.isdigit():
            return True
    return False


def split_text_with_numbers_as_delimiters(text):
    # 去除只有数字的页脚
    if text.isdigit():
        return [text], []

    text = text.replace(".", "")
    delimiters = re.findall(r'\d+', text)
    # 使用负向预查来排除包含 "12V" 的数字
    delimiters = [number for number in delimiters if not (re.search(r'12V', number) or re.search(r'360°', number))]
    
    if delimiters:
        pattern = '|'.join(map(re.escape, delimiters))
        
        # 使用正则表达式切分文本
        parts = re.split(pattern, text)
        
        # 去除空字符串
        parts = [part.strip() for part in parts if part.strip() ]
        delimiters = [int(delimiter.strip()) for delimiter in delimiters if delimiter.strip() ]
        return parts, delimiters
    else:
        # 如果没有找到数字作为分隔符，则返回原始文本
        return [text], delimiters
    

def parse_page_of_content(pdf_path='data/QA.pdf'):
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    documents = text_splitter.split_documents(pdf_docs)
    table_of_content = dict()
    page_index = None
    ## 1-7 页为目录
    for page in documents[1:7]:
        sents = page.page_content.split('\n')
        for sent in sents:
            if "目录" in sent:
                continue
            elif not contains_digit(sent):
                if page_index != None:
                    main_chapter_name = main_chapter_name.replace("V电源", "车载12V电源").replace("°全景影像", "360°全景影像")
                    table_of_content[main_chapter_name] = page_index
                page_index = defaultdict(list)
                main_chapter_name = sent
            else:
                sub_sections, page_ids = split_text_with_numbers_as_delimiters(sent)
                if  len(sub_sections) != len(page_ids):
                    print(f"Not matched chapter name and page_num, source: {sent}")
                for sub_section, page in zip(sub_sections, page_ids):
                    page_index[sub_section].append(page)
    return documents, table_of_content

def find_first_key_geq(d, x):
    for key, value in reversed(d.items()):
        if x >= value:
            return key
    return "None"  # 返回 None 如果没有找到满足条件的键

def parse_section_doc(documents, all_key_word):
        
    all_text = ""

    for doc in documents:
        # 页码删除
        page_id = doc.metadata['page'] + 1
        content = doc.page_content.replace(str(page_id), "")

        new_lines = []
        tmp = ""
        ## 修正换行
        for line in content.split("\n"):
            line = line.strip()
            tmp += line
            if line.endswith("。"):
                new_lines.append(tmp)
                tmp = ""
            elif line in all_key_word:
                # 添加 <sub_section> 标签
                line = "\n<sub_section>" + line
                if len(new_lines) > 0:
                    new_lines[-1] = new_lines[-1] + "\n<\sub_section>"
                new_lines.append(line)
                tmp = ""

        new_lines.append(tmp)
        
        # section 页眉删除
        section_name = find_first_key_geq(section_start_page_id, page_id)
        if new_lines[0].startswith(section_name):
            new_lines[0] = new_lines[0].replace(section_name, "")

        content = "\n".join(new_lines)

        content = content.replace("警告！", "<SEP>警告:\n")
        content = content.replace("注意！", "<SEP>注意:\n")
        content = content.replace("说明！", "<SEP>说明:\n")
        # all_text += f"\n<PAGE_SEP> page_id:{page_id}\n" + content
        all_text += content