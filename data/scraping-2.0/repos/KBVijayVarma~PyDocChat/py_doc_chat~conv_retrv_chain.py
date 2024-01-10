from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from py_doc_chat.model import code_llama
from py_doc_chat.db import db_from_dir


def prompt_format(system_prompt, instruction):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def model_prompt():
    system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
    Read the given context before answering questions and think step by step. If you can not answer a user question based on the provided context, inform the user. Do not use any other information for answering user"""
    instruction = """
    Context: {context}
    User: {question}"""
    template = prompt_format(system_prompt, instruction)
    return template


def custom_que_prompt():
    que_system_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question and Give only the standalone question as output in the tags <question> and </question>.
    """

    instr_prompt = """Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    que_prompt = prompt_format(que_system_prompt, instr_prompt)
    return que_prompt

# def memory_model_prompt():
#     mem_system_prompt = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

#     EXAMPLE
#     Current summary:
#     The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

#     New lines of conversation:
#     Human: Why do you think artificial intelligence is a force for good?
#     AI: Because artificial intelligence will help humans reach their full potential.

#     New summary:
#     The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
#     END OF EXAMPLE

#     """

#     instr_prompt = """Current summary:
#     {summary}

#     New lines of conversation:
#     {new_lines}

#     New summary:"""

#     full_memory_prompt = prompt_format(mem_system_prompt, instr_prompt)
#     return full_memory_prompt


def conv_retrv_chain(retriever=None):
    model_template = model_prompt()
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=model_template
    )
    question_prompt = PromptTemplate.from_template(custom_que_prompt())
    llm = code_llama()
    if not retriever:
        retriever = db_from_dir()
    # memory = ConversationSummaryMemory(
    #     llm=llm,
    #     memory_key='chat_history',
    #     return_messages=True)
    memory = ConversationBufferMemory(
        # llm=llm,
        memory_key='chat_history',
        return_messages=True)
    # memory.prompt.template = memory_model_template
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
                                               memory=memory,
                                               chain_type='stuff',
                                               verbose=True,
                                               combine_docs_chain_kwargs={
                                                   'prompt': QA_CHAIN_PROMPT},
                                               condense_question_prompt=question_prompt)
    return qa
