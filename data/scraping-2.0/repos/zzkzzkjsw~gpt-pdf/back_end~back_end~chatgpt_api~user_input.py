from .utils.chatgpt_utils import num_tokens_from_messages
from .chat_model import ChatModel
import openai
import numpy as np
from typing import List
from pathlib import Path

chatModel = ChatModel()
chatModel.add_system_message("You are a helpful assistant.")


def handle_user_input(request_data):

    # if("parentMessageId" not in request_data):
    #     chatModel.clear_conversation()
        
    user_input = request_data.get("prompt")

    chatModel.add_user_question(user_input)

    print(f"Conversation history token count before answer: {chatModel.num_tokens}")
    while (chatModel.num_tokens + chatModel.max_response_tokens >= chatModel.token_limit):
        chatModel.trim_conversation()
    print(f"Conversation length: {len(chatModel.conversation)}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages = chatModel.conversation,
        temperature=.7,
        max_tokens=chatModel.max_response_tokens,
    )

    chatModel.add_assistant_answer(response['choices'][0]['message']['content'])
    print(response)
    post_response = chatModel.num_tokens
    print(f"Post response total token count: {post_response}")
    
    assert(post_response == num_tokens_from_messages(chatModel.conversation))


    response_data = {}
    response_data['detail'] = response
    response_data['role'] = 'assistant'
    response_data['id'] = response['id']
    response_data['parentMessageID'] = ''
    response_data['text'] = response['choices'][0]['message']['content']
    return response_data

def vector_similarity(x: List[float], y: List[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def get_embedding(text):
    EMBEDDING_DIM = 1536
    model = "text-embedding-ada-002"
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result['data'][0]['embedding']

def handle_user_file_input(request_data,file_text_embedding_list,file_text_list):

    user_input = request_data.get("prompt")
    user_input_embedding = get_embedding(user_input)
    # score, doc_index
    document_similarities = sorted([
        (vector_similarity(user_input_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in enumerate(file_text_embedding_list)
    ], reverse=True, key=lambda x: x[0])

    CONTEXT_TOKEN_LIMIT = 2000

    for (score,doc_index) in document_similarities:
        ctx = ""
        next = ctx + "\n" + file_text_list[doc_index]
        if len(next)>CONTEXT_TOKEN_LIMIT:
            break
        ctx = next
    if len(ctx) == 0:
        return u""


    print(__file__,ctx)
    #
    prompt = u"".join([
    u"Answer the question base on the context, answer in the same language of question\n\n"
    u"Context:" + ctx + u"\n\n"
    u"Question:" + user_input + u"\n\n"
    u"Answer:"
                    ])
    chatModel.add_user_question(prompt)

    print(f"Conversation history token count before answer: {chatModel.num_tokens}")
    while (chatModel.num_tokens + chatModel.max_response_tokens >= chatModel.token_limit):
        chatModel.trim_conversation()
    print(f"Conversation length: {len(chatModel.conversation)}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages = chatModel.conversation,
        temperature=.7,
        max_tokens=chatModel.max_response_tokens,
    )

    chatModel.add_assistant_answer(response['choices'][0]['message']['content'])
    print(response)
    post_response = chatModel.num_tokens
    print(f"Post response total token count: {post_response}")
    
    assert(post_response == num_tokens_from_messages(chatModel.conversation))


    response_data = {}
    response_data['detail'] = response
    response_data['role'] = 'assistant'
    response_data['id'] = response['id']
    response_data['parentMessageID'] = ''
    response_data['text'] = response['choices'][0]['message']['content']

    return response_data