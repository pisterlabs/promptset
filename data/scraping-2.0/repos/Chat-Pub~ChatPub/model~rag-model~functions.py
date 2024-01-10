from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from prompts import *
import openai

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

class BufferMemory():
    def __init__(self, buffer_size):
        # chat history storage
        self.chat_history = []
        self.buffer_size = buffer_size

    def save(self, curr_chat: dict):
        # save the current chat history in memory

        # chat format:
        # {'Human':str,
        #   'AI': str}
        self.chat_history.append(curr_chat)

        if len(self.chat_history) > self.buffer_size:
            self.chat_history = self.chat_history[-self.buffer_size:]
    
    def load(self) -> str:
        chat_history_str = ""

        for chat in self.chat_history:
            for key, value in chat.items():
                chat_history_str += f'{key}: {value}\n'
        
        return chat_history_str

    def clear(self):
        self.chat_history = []
    
def condense_question(model_name, memory, question):
    chat_history = memory.load()

    messages = [{"role":"system", 
                 "content":condense_question_system},
                 {"role":"user",
                  "content":condense_question_user_template.format(chat_history=chat_history, question=question)}]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature = 0
    )

    return response.choices[0]['message']['content']


def set_index(data_list):
    # dim size
    d = 768

    # set index
    index = faiss.IndexFlatL2(d)
    Index = faiss.IndexIDMap2(index)

    # add the items into Index
    for item in data_list:
        id = item['YP']
        content = f"제목: {item['title']}\n"
        content += f"지원내용: {item['contents']['short_description']}\n"

        qualification = item['contents']['qualification']
        for key, value in qualification.items():
            if value != '-':
                content += f'{key}: {value}\n'

        content_vector = model.encode([content])
        vector_id = np.array([id], dtype='int64')
        Index.add_with_ids(content_vector, vector_id)
    
    return Index


def get_context(index, query, context_cnt, data_list):
    query_vector = model.encode([query])
    distance, index = index.search(query_vector, context_cnt)
    
    context= ""
    title_url_lists = []
    # idx shape: [query 개수, k]
    # shape: (1, k)
    for i in index[0]:
        title_url_item = {}
        for key, value in data_list[i].items():
            if key == 'title':
                context += f"제목: {value}\n"

                title_url_item['title'] = value
            
            elif key == 'contents':
                context += f"설명 요약: {value['short_description']}\n"
                context += f"요약: {value['summary']}\n"
                context += f"자격요건: {value['qualification']}\n"
                context += f"신청방법: {value['methods']}\n"

                title_url_item['url'] = value['url']

        title_url_lists.append(title_url_item)
        
        context += "##############################\n"
    
    return context, title_url_lists


def compute_similarity(response, title):
    preprocessed_response = response.replace('"','').replace("'",'').replace('\n', ' ').replace('  ', ' ')
    preprocessed_title = title.replace('"','').replace("'",'').replace('\n', ' ').replace('  ', ' ')
    tokenized_response = set(preprocessed_response.split(' '))
    tokenized_title = set(preprocessed_title.split(' '))

    return len(tokenized_title.intersection(tokenized_response)) / len(tokenized_title)


def get_response(model_name, context, query):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role":"system", "content":chat_system_prompt},
                    {"role":"user", "content":chat_user_prompt_template.format(context=context, query=query)}],
        temperature = 0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response['choices'][0]['message']['content']

def get_valid_references(response, title_url_lists):
    references = ''

    for title_url_item in title_url_lists:
        title = title_url_item['title']
        url = title_url_item['url']
        
        sim = compute_similarity(response=response, title=title)

        if sim >= 0.5:
            if references != "":
                references += '\n'
            references += f'제목: {title}\n링크:{url}\n'

    return references


    