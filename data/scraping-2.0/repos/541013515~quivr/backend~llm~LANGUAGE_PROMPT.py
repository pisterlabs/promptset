from langchain.prompts.prompt import PromptTemplate

_template = """根据以下对话和跟进问题，用提问语言回答跟进问题。如果你不知道答案，只需说不知道，不要试图编造答案。

聊天记录：
{chat_history}
跟进问题输入：{question}
独立问题："""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question in the language of the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )