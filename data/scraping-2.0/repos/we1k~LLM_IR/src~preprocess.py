import os
import re
import json
from collections import defaultdict
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma

from src.text_splitter import ChineseRecursiveTextSplitter
from src.utils import save_docs_to_jsonl, load_docs_from_jsonl, load_embedding_model

MAX_KEYWORD_LEN = 13
DELIMITER = ['，', ',', '。', '；', '–', '：', '！', '-', '、', '■', '□', '℃',
             '.', '•']

def save_docs_to_jsonl(array, file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(json.dumps(doc.dict(), ensure_ascii=False) + '\n')

def contains_chinese_characters(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 匹配汉字的 Unicode 范围
    match = pattern.search(text)
    return match is not None

def get_keywords(file_path='pdf_output/trainning_data.outline'):
    # File path to the outline document
    # Read the entire file content
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Use regular expression to find all instances of chapter names in <a> tags
    chapter_details = re.findall(
    r'<a class="l"[^>]*data-dest-detail=\'\[(\d+),[^\]]+\]\'>(.*?)\s*</a>',
    file_content
    )

    chapter_to_number_dict = {detail[1].strip(): int(detail[0]) for detail in chapter_details}
    chapter_names = [k.replace("&amp;", "&").strip() for k, v in chapter_to_number_dict.items()]
    return chapter_names


def build_sections(keywords, max_sentence_len=29):
    section_docs = []
    sections = defaultdict(str)
    chapter_name = ""
    tmp = ""

    with open("data/all.txt", 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            # 去除目录
            if ". . . ." in line or "目录" in line or "...." in line:
                continue
            if line.strip() in keywords:
                if chapter_name != "":
                    sections[chapter_name] += "<SEP>" + tmp
                    tmp = ""
                chapter_name = line.strip()
            else:
                tmp += line
    
    with open("data/section.json", 'w', encoding='UTF-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)

    sections[chapter_name] = tmp
    for chapter_name, text in sections.items():
        subsection_dict = {}
        text = text.replace("点击\n", "点击")
        text = text.replace("\n-", "-")
        text = text.replace("\n“\n", "")
        text = text.replace("\n”\n", "")
        text = text.replace("\\", "")
        text = re.sub(r"\n\d+km/h\n", "", text)
        text = text.replace("的\n", "的")
        sentences = text.split('\n')
        
        keyword = chapter_name
        cur_chunk = chapter_name + "\n"
        for sentence in sentences:
            sentence = sentence.strip("<SEP>").replace('"', '').replace(" ", "")
            if len(sentence) == 0 or sentence.isdigit():
                continue
            # 大概率是目录
            # 可能包含章节数字 1.1 标题
            elif re.match(r"^\d*(\.\d+)?.*$", sentence) and not any(it in sentence for it in DELIMITER) and not sentence.startswith("0") and 1< len(sentence) <= MAX_KEYWORD_LEN:
                if cur_chunk.strip("\n") != keyword:
                    subsection_dict[keyword] = cur_chunk
                keyword = sentence
                cur_chunk = sentence + "\n"
            # 拼接上下句子
            elif len(sentence) >= max_sentence_len - 1 or ("，" in sentence and not sentence.endswith("。")):
                cur_chunk += sentence
            # 换行后第一个字符是分隔符
            elif any(sentence.startswith(it) for it in DELIMITER):
                cur_chunk = cur_chunk.strip("\n") + sentence + "\n"
            else:
                cur_chunk += sentence + "\n"

        # adding last chunk
        if cur_chunk.strip("\n") != keyword:
            subsection_dict[keyword] = cur_chunk

        for subkeyword, text_chunk in subsection_dict.items():
            if len(text_chunk.strip("<SEP>")) > 0 and not text_chunk.isalpha() > 0:
                # skip special char
                text_chunk = text_chunk.replace("<SEP>", "")
                # skip too short section (maybe a table content)
                # or section name didn't contain chinese characters
                if len(text_chunk.replace("\n", "")) - len(subkeyword) < 5 or not contains_chinese_characters(subkeyword):
                    continue
                section_docs.append(Document(page_content=text_chunk, metadata={"keyword": chapter_name, "subkeyword": subkeyword}))
        
    return section_docs

def preprocess(embeddings, max_sentence_len=20):
    keywords = get_keywords()
    # print(keywords)
    with open("data/raw.txt", 'r', encoding='UTF-8') as f:
        text = f.read()

    pages = re.split(r'!\[\]\(.+?\)', text)
    # 去掉页眉和页码
    for i in range(len(pages)):
        lines, idx = pages[i].split("\n"), 0
        lines = [line for line in lines if len(line.strip()) > 0]
        while len(lines) > 0 and not contains_chinese_characters(lines[-1]):
            lines.pop(-1)
        
        while 0 <= idx < len(lines) and lines[idx].strip().isdigit():
            idx += 1
        pages[i] = "\n".join(lines[idx+2:])
        # pages[i] = re.sub(rf'^.*?\n{i}\n', "", pages[i], flags=re.DOTALL)

        # 去掉图片的编号
        pages[i] = re.sub(r"[A-Za-z0-9]+-[A-Za-z0-9]+\n", "", pages[i])
        
        pages[i] = pages[i]

    all_text = "".join(pages).replace("\n\n", "\n")
    with open("data/all.txt", 'w', encoding='UTF-8') as f:
        f.write(all_text)

    section_docs = build_sections(keywords, max_sentence_len)

    # section_docs_tmp =  load_docs_from_jsonl("doc/section_docs_1.jsonl")
    # section_docs += section_docs_tmp
    section_docs = load_docs_from_jsonl("doc/section_docs.jsonl")

    all_keywords = [doc.metadata["keyword"] for doc in section_docs] + [doc.metadata["subkeyword"] for doc in section_docs]
    all_keywords = list(set(all_keywords))

    with open("data/keywords.txt", 'w', encoding='UTF-8') as f:
        f.write("\n".join(all_keywords))
        

    db = FAISS.from_documents(section_docs, embeddings)
    db.save_local("vector_store/section_db")


    # index_db = FAISS.from_texts(all_keywords, embeddings)
    # index_db.save_local("vector_store/index_db")
    # index_db = FAISS.load_local('vector_store/index_db', embeddings)

    # sentence cut
    chunk_size = 120
    chunk_overlap = 20
    sentence_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    for doc in section_docs:
        doc.page_content = doc.page_content.replace(" ", "")
        
    sent_docs = sentence_splitter.split_documents(section_docs)
    
    # adding index and combine
    cur_doc_content = ""
    clean_sent_docs = []
    for doc in sent_docs:
        cur_doc_content += doc.page_content
        # TODO : 50 is the hyperparameter, and using doc.page_content instead of cur_doc_content may be a bug?
        # doc.page_content > 50 means the doc is a complete sentence, but cur_doc_content is not
        if len(doc.page_content) >= 50:
            doc.page_content = cur_doc_content
            # doc.page_content = doc.page_content.replace(" ", "")
            doc.page_content = doc.page_content.replace("<SEP>", "")
            doc.page_content = doc.page_content.replace("•", "")
            doc.page_content = doc.page_content.replace("□", "")
            doc.page_content = doc.page_content.strip("。\n")
            doc.metadata['index'] = len(clean_sent_docs)
            clean_sent_docs.append(doc)
            cur_doc_content = ""
    sent_docs = clean_sent_docs


    save_docs_to_jsonl(sent_docs, "doc/sent_docs.jsonl")
    sent_db = FAISS.from_documents(sent_docs, embeddings)
    sent_db.save_local("vector_store/sentence_db")

if __name__ == '__main__':
    embeddings = load_embedding_model("stella", True)
    preprocess(embeddings)