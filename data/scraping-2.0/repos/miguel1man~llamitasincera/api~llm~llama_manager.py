import os
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))

template_data = """
Pregunta:
{question}
Respuesta: 
Responde en idioma español únicamente utilizando estos datos como respuesta, no inventes hechos, solo utiliza estos datos:
{answer_data}"""


def llm_vector_similarity(question, answer_data, model_name, template=template_data):
    try:
        prompt = PromptTemplate(
            template=template,
            input_variables=["answer_data", "question"],
        )

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            callback_manager=callback_manager,
            max_tokens=4096,
            n_ctx=2048,
            model_path=f"{parent_dir}/models/{model_name}",
            temperature=0,
            verbose=True,
            streaming=True,
        )

        llm_chain = LLMChain(prompt=prompt, llm=llm)

        inputs = {
            "answer_data": answer_data,
            "question": question,
        }

        return llm_chain.run(inputs)

    except Exception as e:
        print(f"Error on llm processing data: {e}")


template_question = """
Pregunta:
{question}
Respuesta: 
Responde en idioma español de manera muy concisa."""


def llm_question(question, model_name, template=template_question):
    try:
        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
        )

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            callback_manager=callback_manager,
            max_tokens=4096,
            n_ctx=2048,
            model_path=f"{parent_dir}/models/{model_name}",
            temperature=0,
            verbose=True,
        )

        llm_chain = LLMChain(prompt=prompt, llm=llm)

        inputs = {
            "question": question,
        }

        return llm_chain.run(inputs)

    except Exception as e:
        print(f"Error on llm processing question: {e}")
