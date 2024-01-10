import re
import tqdm
import json
import spacy
import PyPDF2
from argparse import ArgumentParser
from src.embeddings import BGEpeftEmbedding
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def extract_page_text(filepath, max_len=256):
    page_content  = []
    spliter = spacy.load("zh_core_web_sm")
    chunks = []
    with open(filepath, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        page_count = 10
        pattern = r'^\d{1,3}'
        for page in tqdm.tqdm(pdf_reader.pages[page_count:]):
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.split('\n')]
            new_text = '\n'.join(raw_text[1:])
            new_text = re.sub(pattern, '', new_text).strip()
            page_content.append(new_text)
            max_chunk_length = max_len  # 最大 chunk 长度

            current_chunk = ""
            if len(new_text) > 10:
                for sentence in spliter(new_text).sents:
                    sentence_text = sentence.text
                    if len(current_chunk) + len(sentence_text) <= max_chunk_length:
                        current_chunk += sentence_text
                    else:
                        chunks.append(Document(page_content=current_chunk, metadata={'page':page_count+1}))
                        current_chunk = sentence_text
                # 添加最后一个 chunk（如果有的话）
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk, metadata={'page':page_count+1}))
            page_count += 1
    cleaned_chunks = []
    i = 0
    while i <= len(chunks)-2: #简单合并一些上下文
        current_chunk = chunks[i]
        next_chunk = chunks[min(i+1, len(chunks)-1)]
        if len(next_chunk.page_content) < 0.5 * len(current_chunk.page_content):
            new_chunk = Document(page_content=current_chunk.page_content + next_chunk.page_content, metadata=current_chunk.metadata)
            cleaned_chunks.append(new_chunk)
            i += 2
        else:
            i+=1
            cleaned_chunks.append(current_chunk)

    return cleaned_chunks


def run_query(args):
    ## pdf -> Doc
    if args.local_run == True:
        filepath = "data/trainning_data.pdf"
    else:
        filepath = "/tcdata/trainning_data.pdf"
    docs = extract_page_text(filepath=filepath, max_len=256)
    # load in embedding model
    if "bge" in args.embedding_model:
        if local_run:
            model_name = "./models/bge-large-zh-v1.5"
        else:
            model_name = "/app/models/bge-large-zh-v1.5"
        embeddings = BGEpeftEmbedding(model_name)
    elif "stella" in args.embedding_model:
        if args.local_run:
            model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
        else:
            model_name = "/app/rerank_model/stella-base-zh-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"} ,
            encode_kwargs={"normalize_embeddings": False})
    elif "gte" in args.embedding_model:
        model_name = "/app/models/gte-large-zh"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"} ,
            encode_kwargs={"normalize_embeddings": False})
        
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(folder_path='./vector', index_name='index_256')

    if args.local_run == True:
        question_path = './data/all_question.json'
    else:
        question_path = "/tcdata/test_question.json"

    with open(question_path, 'r', encoding='utf-8') as f:
        question_list = json.load(f)

    answers=[]
    for i, line in enumerate(question_list):
        # print(f"question {i}:", line['question'])
        search_docs = db.similarity_search(line['question'], k=args.max_num_related_str)
        # print(search_docs)
        related_str = []
        for doc in search_docs:
            related_str.append(doc.page_content)
        sample = {"question": line['question'], "related_str": related_str, "keyword": ""}
        answers.append(sample)


    with open(f"result/related_str.json", 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--max_num_related_str", default=5, type=int)
    parser.add_argument("--local_run", action="store_true")
    parser.add_argument("--embedding_model", default="stella")
    args = parser.parse_args()
    # bge // stella // gte
    run_query(args)