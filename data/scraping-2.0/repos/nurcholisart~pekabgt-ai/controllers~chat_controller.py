import tempfile
import requests
import os

from fastapi.responses import JSONResponse

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import logging

logging.getLogger("openai").setLevel(logging.DEBUG)  # logging.INFO or logging.DEBUG

_condense_template = """Please rephrase the following question into good grammar.
Please respond in same language.

Question:
{question}

Rephrased question:"""

condense_prompt = PromptTemplate.from_template(_condense_template)


class ChatController:
    def call(
        self,
        api_key: str,
        faiss_url: str,
        pkl_url: str,
        question: str,
        system_prompt: str,
        chat_history: list[tuple] = [],
    ) -> dict:
        with get_openai_callback() as cb:
            try:
                context = """                
                Context:
                {context}

                Question:
                {question}
                
                Helpful Answer:"""

                template = system_prompt + "\n\n" + context

                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"],
                )

                result = {}

                with tempfile.TemporaryDirectory() as tempdirname:
                    resp1 = requests.get(faiss_url, allow_redirects=True)

                    with open("index.faiss", "wb") as file:
                        file.write(resp1.content)

                    resp2 = requests.get(pkl_url, allow_redirects=True)
                    with open("index.pkl", "wb") as file:
                        file.write(resp2.content)

                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    vectorstore = FAISS.load_local(".", embeddings)

                    chain = ConversationalRetrievalChain.from_llm(
                        llm=OpenAI(
                            openai_api_key=api_key,
                            temperature=0,
                            max_tokens=-1,
                            verbose=True,
                        ),
                        retriever=vectorstore.as_retriever(
                            search_kwargs={
                                "k": 2,
                                "search_type": "mmr",
                                "search_distance": 0.9,
                            }
                        ),
                        return_source_documents=False,
                        condense_question_prompt=condense_prompt,
                        combine_docs_chain_kwargs={"prompt": prompt},
                        verbose=True,
                    )

                    raw_result = chain(
                        {
                            "question": question,
                            "chat_history": chat_history,
                            "meta": {},
                        }
                    )

                    chat_history.append((raw_result["question"], raw_result["answer"]))

                    result = {
                        "success": True,
                        "question": raw_result["question"],
                        "answer": raw_result["answer"],
                        "chat_history": raw_result["chat_history"],
                        "source_documents": [],
                        "meta": {
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_tokens": cb.total_tokens,
                            "total_cost": cb.total_cost,
                            "success_requests": cb.successful_requests,
                        },
                    }

                    return JSONResponse(result, status_code=200)
            except Exception as e:
                print(e)
                result = {
                    "success": False,
                    "message": f"Request to AI service failed. {str(e)}",
                    "meta": {
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost,
                        "success_requests": cb.successful_requests,
                    },
                }

                return JSONResponse(result, status_code=500)
