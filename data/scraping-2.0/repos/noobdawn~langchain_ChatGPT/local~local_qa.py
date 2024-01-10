from typing import List, Tuple
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import openai
from configs.model_config import *
import shutil
import json

# 获取问题答案对
def extract_qa_from_txt(file_path):
    questions = []
    answers = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Q：') or line.startswith('Q:'):
            question = line[2:]
            i += 1
            while i < len(lines) and not (lines[i].startswith('A：') or lines[i].startswith('A:')):
                question += '\n' + lines[i].strip()
                i += 1
            questions.append(question)

        elif line.startswith('A：') or line.startswith('A:'):
            answer = line[2:]
            i += 1
            while i < len(lines) and not (lines[i].startswith('Q：') or lines[i].startswith('Q:')):
                answer += '\n' + lines[i].strip()
                i += 1
            answers.append(answer)

        else:
            i += 1

    return questions, answers


# delete the start part such as "1."
def remove_digital_start(question : str) -> str:
    if question[0].isdigit():
        return question[question.find('.') + 1:].strip()
    else:
        return question


# 获取同义问句
def get_synonymous_question(question : str) -> list:
    openai.api_key = api_key
    prompt = prompt_base_templet + question
    response = openai.Completion.create(
        engine=gpt_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n = output_num
    )
    generate_text = response.choices[0].text.strip()
    questions_array = generate_text.split('\n')
    # 清理空句子
    questions_array = [question for question in questions_array if question != '']
    # 清理数字开头的句子
    questions_array = [remove_digital_start(question) for question in questions_array]
    return questions_array


# 从原文获得问题答案对
def get_questions_and_answers_dict_from_txt(file_list : list) -> dict:
    result = {}
    failed_files = []
    for file in file_list:
        try:
            filename = os.path.split(file)[-1]
            question_dict = {}
            questions, answers = extract_qa_from_txt(file)
            for i in range(len(questions)):
                original_question = questions[i]
                oq_dict = {}
                oq_dict['answer'] = answers[i]
                synonymous_questions = get_synonymous_question(original_question)
                oq_dict['synonymous'] = synonymous_questions
                question_dict[original_question] = oq_dict
            result[filename] = question_dict
        except Exception as e:
            failed_files.append(file)
    return result, failed_files


# 从字典获得问题答案对
def get_questions_and_answers_dict_from_dict(file_list : list) -> dict:
    result = {}
    failed_files = []
    for file in file_list:
        try:
            filename = os.path.split(file)[-1]
            with open(file, 'r', encoding='utf-8') as f:
                qa_dict = json.load(f)
            result[filename] = qa_dict
        except Exception as e:
            failed_files.append(file)
    return result, failed_files


# 问题答案对字典转化为Document
def qa_dict_to_docs(qa_dict : dict) -> List[Document]:
    docs = []
    for reference in qa_dict.keys():
        question_dict = qa_dict[reference]
        for original_question in question_dict.keys():
            oq_dict = question_dict[original_question]
            synonymous_questions = oq_dict['synonymous']
            for synonymous_question in synonymous_questions:
                doc = Document(page_content=synonymous_question, metadata={"source" : reference,
                                                          "original" : original_question,
                                                          "answer" : oq_dict['answer']})
                docs.append(doc)
            original_doc = Document(page_content=original_question, metadata={"source" : reference,
                                                                              "original" : original_question,
                                                                              "answer" : oq_dict['answer']})
        docs.append(original_doc)
    return docs


class LocalQA:
    def __init__(self,
                 embedding_model : str,
                 embedding_device : str, 
                 top_k : int) -> None:
        self.embeddings = None
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.top_k = top_k


    def copy_files(self, 
                   file_path : list, 
                   target_dir : str):
        new_files = []
        updated_files = []
        for file in file_path:
            target_filename = os.path.join(target_dir, os.path.split(file)[-1])
            if os.path.exists(target_filename):
                updated_files.append(target_filename)
            else:
                new_files.append(target_filename)
            shutil.copyfile(file, target_filename)
        return new_files, updated_files
    

    def build_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[self.embedding_model],
                                                model_kwargs={'device': self.embedding_device})


    def update_knowledge_base_new(self,
                              kb_name : str,
                              file_path : str or list):
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[self.embedding_model],
                                                model_kwargs={'device': self.embedding_device})
        # 文件夹判断
        vs_path = os.path.join(VS_ROOT_PATH, get_pinyin(kb_name))
        txt_path = os.path.join(UP_FILE_PATH, kb_name)
        dict_path = os.path.join(UP_DICT_PATH, kb_name)
        img_path = os.path.join(UP_IMG_PATH, kb_name)
        if not os.path.isdir(vs_path):
            os.makedirs(vs_path)
        if not os.path.isdir(txt_path):
            os.makedirs(txt_path)
        if not os.path.isdir(dict_path):
            os.makedirs(dict_path)
        if not os.path.isdir(img_path):
            os.makedirs(img_path)
        # 划分字典和原文
        if isinstance(file_path, str):
            file_path = [file_path]
        txt_files = [file for file in file_path if file.endswith('.txt')]
        dict_files = [file for file in file_path if file.endswith('.dict')]
        img_files = [file for file in file_path if not file.endswith('.txt') and not file.endswith('.dict')]
        # 复制原文到file_path下
        new_txt, updated_txt = self.copy_files(txt_files, txt_path)
        # 提取问题答案对并保存
        new_qa_dict, failed_files = get_questions_and_answers_dict_from_txt(new_txt + updated_txt)
        new_files = [x for x in new_txt if x not in failed_files]
        updated_files = [x for x in updated_txt if x not in failed_files]     
        # 复制图片到img_path下
        new_img, updated_img = self.copy_files(img_files, img_path)
        new_files += [x for x in new_img if x not in failed_files]
        updated_files += [x for x in updated_img if x not in failed_files]
        # 复制字典到dict_path下
        new_dict, updated_dict = self.copy_files(dict_files, dict_path)
        # 提取字典里的问题答案对并保存
        new_qa_dict_dict, failed_files = get_questions_and_answers_dict_from_dict(new_dict + updated_dict)
        new_files += [x for x in new_dict if x not in failed_files]
        updated_files += [x for x in updated_dict if x not in failed_files]
        # 合并
        new_qa_dict.update(new_qa_dict_dict)
        # 获得之前的字典
        old_qa_dict = {}
        if os.path.exists(os.path.join(dict_path, "dict.dict")):
            with open(os.path.join(dict_path, "dict.dict"), 'r', encoding='utf-8') as f:
                old_qa_dict = json.load(f)
        # 更新字典
        for new_ref in new_qa_dict.keys():
            old_qa_dict[new_ref] = new_qa_dict[new_ref]
        # 保存字典
        with open(os.path.join(dict_path, "dict.dict"), 'w', encoding='utf-8') as f:
            json.dump(old_qa_dict, f, ensure_ascii=False, indent=4)
        # 保存向量存储
        docs = qa_dict_to_docs(old_qa_dict)
        faiss = FAISS.from_documents(docs, self.embeddings)
        faiss.save_local(vs_path)
        return new_files, updated_files, failed_files


    def query_answer_with_score(self,
                                name : str,
                                query : str,
                                top_k : int) -> List[Tuple[str, float]]:
        vs_path = os.path.join(VS_ROOT_PATH, name)
        if not os.path.isdir(vs_path):
            return []
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[self.embedding_model],
                                                model_kwargs={'device': self.embedding_device})
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        possible_answers = vector_store.similarity_search_with_score(query, top_k)
        return possible_answers


if __name__ == "__main__":
    # 测试回答
    qa = LocalQA("text2vec", "cpu", 5)
    possible_questions = qa.query_answer_with_score("知识库", "问题", 5)
    original_questions = []
    answers = []
    for question, score in possible_questions:
        if question.metadata["source"] not in original_questions:
            original_questions.append(question.metadata["source"])
            answers.append((question, score))
    answers.sort(key=lambda x: x[1], reverse=True)
    print("您要找的答案可能是：")
    for question, score in answers:
        print(f"{question.metadata['answer']}（{question.metadata['original']} - {question.metadata['source']}） 相关性：{score}")

    # 测试更新知识库
    qa = LocalQA("text2vec", "cpu", 5)
    new_files, updated_files, failed_files = qa.update_knowledge_base_new("知识库", ["doc/aaa.txt", "doc/bbb.dict"])
    print(new_files, updated_files, failed_files)