# from langchain.chains import RetrievalQA
from langchain_ChatGLM.retrievalchain import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain_ChatGLM.chatglm_llm import ChatGLM
import sentence_transformers
import torch
import os
from langchain.document_loaders import CSVLoader
from langchain.chains.llm import LLMChain
import copy



# Global Parameters
VECTOR_SEARCH_TOP_K = 2
LLM_MODEL = "chatglm-6b"
LLM_HISTORY_LEN = 5
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

llm_model_dict = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b": "THUDM/chatglm-6b",
}

    
def init_cfg(LLM_MODEL, LLM_HISTORY_LEN, V_SEARCH_TOP_K=2):
    global chatglm, embeddings, VECTOR_SEARCH_TOP_K
    VECTOR_SEARCH_TOP_K = V_SEARCH_TOP_K

    chatglm = ChatGLM(url="http://10.102.32.99:9101/")
    # chatglm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL])
    chatglm.history_len = LLM_HISTORY_LEN


def get_knowledge_based_answer(query, chat_history=[]):
    global chatglm

    prompt_template = """基于以下已知内容，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分。

已知内容:
{context}

问题:
{question}"""

    prompt_template = """基于以下已知内容，尽可能地简洁和专业的来回答用户的问题，不允许在答案中添加编造的成分。

已知内容:
{context}

问题:
{question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chatglm.history = chat_history


    condese_propmt_template = """任务: 给一段对话和一个后续问题，将后续问题改写成一个独立的问题。确保问题是完整的，没有模糊的指代。
    ----------------
    聊天记录：
    {chat_history}
    ----------------
    后续问题：{question}
    ----------------
    改写后的独立、完整的问题："""

    # condese_propmt_template = """任务: 基于以下用户与问答助手的对话记录，将用户的后续问题改写成一个独立的问题，确保问题是完整的，没有模糊的指代。
    # ----------------
    # 对话记录：
    # {chat_history}
    # ----------------
    # 用户的后续问题：{question}
    # ----------------
    # 改写后的独立、完整的问题："""
    new_question_prompt = PromptTemplate.from_template(condese_propmt_template)
    question_generator = LLMChain(llm=chatglm, prompt=new_question_prompt)
    # new_question = question_generator.run(question=query, chat_history=chat_history)
    # print("new_question:",new_question)
    new_question = query

    # chis = copy.deepcopy(chat_history)

    # for i in range(len(chis)):
    #     # 在第一个数据前拼接“问答助手：”
    #     chis[i][0] = "用户：" + chis[i][0]
    #     # 在第二个数据前拼接“用户：”
    #     chis[i][1] = "问答助手：" + chis[i][1]


    if len(chat_history) > 0:
        new_question = question_generator.run(question=query, chat_history=chat_history)
    print("new_question:",new_question)

    knowledge_chain = RetrievalQA.from_llm(
        llm=chatglm,
        prompt=prompt
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

    knowledge_chain.return_source_documents = True

    result = knowledge_chain({"query": new_question})
    # print("result:", result)
    chatglm.history[-1][0] = new_question
    new_list = [lst for lst in chatglm.history if lst[0] is not None]
    chatglm.history = new_list
    return result, chatglm.history


# if __name__ == "__main__":
init_cfg(LLM_MODEL, LLM_HISTORY_LEN, VECTOR_SEARCH_TOP_K)
# vector_store = None
# # while not vector_store:
# #     filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
# #     vector_store = init_knowledge_vector_store(filepath)
# loader1 = CSVLoader(file_path='./datasets/处治方案知识库.csv')
# loader2 = CSVLoader(file_path='./datasets/Numerical Case Library.csv')
# loader3 = CSVLoader(file_path='./datasets/Text Case Library.csv')
# docs=[]
# # documents1.extend(loader1.load())
# docs.extend(loader1.load())
# docs.extend(loader2.load())
# docs.extend(loader3.load())

# vector_store = Chroma.from_documents(docs, embeddings, persist_directory='./store')
# vector_store = FAISS.from_documents(docs, embeddings)
# history = []