from typing import Tuple
from langchain.chat_models import AzureChatOpenAI
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain


def get_templates() -> Tuple[PromptTemplate, PromptTemplate]:

    question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    docs_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:"""

    question_prompt = PromptTemplate.from_template(question_template)
    combine_docs_prompt = PromptTemplate.from_template(docs_template)

    return question_prompt, combine_docs_prompt


def run_conversational_chain(question, history, retriever):
    llm = AzureChatOpenAI(deployment_name="gpt-35", temperature=0)

    def get_chat_history(inputs) -> str:
        res = []
        for input in inputs:
            prefix = "AI Assistant" if input["type"] == "bot" else "Client"
            res.append(f"{prefix}:{input['message']}")
        return "\n".join(res)

    line = "=" * 42
    document_separator = "\n\n" + line + " New Doc " + line + "\n\n"
    question_prompt, combine_docs_prompt = get_templates()

    support_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=question_prompt,
        return_source_documents=False,
        get_chat_history=get_chat_history,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": combine_docs_prompt, "document_separator": document_separator}
    )

    response = support_qa.run({"question": question, "chat_history": history})
    return response