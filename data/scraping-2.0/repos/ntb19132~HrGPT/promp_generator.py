import tiktoken

from services.openai import OPENAI_MODEL, get_embedded

import numpy as np

from services.pinecone import search

ENCODING = tiktoken.encoding_for_model(OPENAI_MODEL)

h_limit = 3800
s_limit = 3300


def count_tokens(message):
    return len(ENCODING.encode(message))


class PromptGenerator:
    prompt_start = f"""Answer the question based on the context below, don’t justify your answers and if the answer cannot be found write "I don't know. Please contact HR for more information"\n\nContext:\n"""

    def __init__(self):
        ...

    def retrieve(self, query):
        # Create embeddings for the query
        response = get_embedded(query)
        # Retrieve from Pinecone
        xq = response['data'][0]['embedding']

        # Get relevant contexts
        contexts = search(xq)
        # Build the initial part of our prompt with the retrieved contexts included

        # Define the end of the prompt
        prompt_end = (
            f"\n\nQuestion: {query}\nAnswer:"
        )

        tmp_cnt_tokens, doc_lim_num = 0, 0
        tmp_cnt_prompt_start = count_tokens(self.prompt_start)
        tmp_cnt_prompt_end = count_tokens(prompt_end)
        for i, ctx in enumerate(contexts):
            tmp_cnt_tokens = tmp_cnt_tokens + count_tokens(ctx['text'])
            if tmp_cnt_tokens + tmp_cnt_prompt_start + tmp_cnt_prompt_end > s_limit:
                doc_lim_num = i - 1
                break

        if doc_lim_num == 0:
            return contexts
        else:
            return contexts[:doc_lim_num + 1]

    def get_prompt(self, question, contexts, chatlogs=[], log_length=3):
        prompt_end = (
            f"\n\nQuestion: {question}\nAnswer:"
        )
        texts = [ctx['text'] for ctx in contexts ]
        prompt = self.prompt_start + "\n\n---\n\n".join(texts) + prompt_end
        # If the length of the context exceeds the log_length, take only the last log_length elements
        if len(chatlogs) > log_length:
            chatlog = chatlogs[-log_length:]
        # If not, take the whole context
        else:
            chatlog = chatlogs

        system_prompt = [{'role': 'system',
                          'content': 'You are an HR assistant that answer only the question that relate to the company. '
                                     'You are not allow to generate or convert any programing language'
                          }]
        tmp_prompt = [{"role": "user", "content": prompt}]

        tmp_msg_tokens = count_tokens(system_prompt[0]["content"])
        tmp_prompt_tokens = count_tokens(tmp_prompt[0]["content"])

        tmp_chat_log_tokens = sum(map(lambda clog: count_tokens(clog['content']), chatlog))

        if tmp_msg_tokens + tmp_prompt_tokens + tmp_chat_log_tokens <= h_limit:
            messages = system_prompt + chatlog + tmp_prompt
            feed_tokens = tmp_msg_tokens + tmp_prompt_tokens + tmp_chat_log_tokens
            print('Tokens Before Feed :', feed_tokens)

        return messages
