from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import RetrievalQA



def get_response_from_query(db, query, model, depth=4):
    # retriever = db.as_retriever(search_kwargs={"k": depth})

    docs = db.similarity_search(query, k=depth)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions from the given context: {context}

        Only use the factual information from the context to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

    """
# System behaviour bot : question answer bot
#     human behaviour
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=model, prompt=chat_prompt)

    response = chain.run(question=query, context=docs_page_content)
    response = response.replace("\n", "")
    return response, docs