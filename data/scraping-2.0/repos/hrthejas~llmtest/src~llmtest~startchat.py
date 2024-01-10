from IPython.display import display, Markdown
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from llmtest import constants


# def get_local_model_llm(model_id=constants.DEFAULT_MODEL_NAME,
#                         use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
#                         set_device_map=constants.SET_DEVICE_MAP,
#                         max_new_tokens=constants.MAX_NEW_TOKENS, device_map=constants.DEFAULT_DEVICE_MAP,
#                         use_simple_llm_loader=False):
#     return llmloader.getLLM(
#         model_id=model_id,
#         use_4bit_quantization=use_4bit_quantization,
#         set_device_map=set_device_map,
#         max_new_tokens=max_new_tokens, device_map=device_map, use_simple_llm_loader=use_simple_llm_loader)


def get_openai_model_llm():
    return ChatOpenAI(model_name=constants.OPEN_AI_MODEL_NAME, temperature=constants.OPEN_AI_TEMP,
                      max_tokens=constants.MAX_NEW_TOKENS)


# def get_local_model_qa_chain(retriever, model_id=constants.DEFAULT_MODEL_NAME,
#                              use_4bit_quantization=constants.USE_4_BIT_QUANTIZATION,
#                              set_device_map=constants.SET_DEVICE_MAP,
#                              max_new_tokens=constants.MAX_NEW_TOKENS, device_map=constants.DEFAULT_DEVICE_MAP):
#     llm = get_local_model_llm(
#         model_id=model_id,
#         use_4bit_quantization=use_4bit_quantization,
#         set_device_map=set_device_map,
#         max_new_tokens=max_new_tokens, device_map=device_map)
#
#     return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def get_openai_model_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def get_local_model_result(qa, final_question):
    return qa(final_question)


def get_chat_gpt_result(qa, final_question):
    return qa(final_question)


def get_prompt():
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate(template=constants.API_QUESTION_PROMPT, input_variables=["context", "question"])
    return prompt


def get_chain(llm, prompt, chain_type="stuff"):
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(llm=llm, chain_type=chain_type, prompt=prompt)
    return chain


# def get_local_model_result_using_chain(llm, question, chain_type="stuff"):
#     from langchain.chains.question_answering import load_qa_chain
#     from langchain.prompts import PromptTemplate
#     prompt = get_prompt()
#     chain = get_chain(llm, chain_type=chain_type, prompt=prompt)
#
#     return qa(final_question)


def get_chat_gpt_result_using_chin(qa, final_question):
    return qa(final_question)


def get_answers(local_qa=None, openai_qa=None, question="list all environments in infoworks", use_prompt=True,
                prompt=constants.API_QUESTION_PROMPT):
    if use_prompt:
        final_question = prompt + '\n' + question
    else:
        final_question = question
    if openai_qa is not None:
        display(Markdown('*OPEN AI Result*'))
        print(get_chat_gpt_result(openai_qa, final_question)['result'])
        print('\n\n\n\n')
    if local_qa is not None:
        display(Markdown('*local llm Result*'))
        print(get_local_model_result(local_qa, final_question)['result'])
