from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

prompt_template = """You are the chatbot of e-shop.
The following pieces of context: Contract offer with E-SHOP named ООО «РусПластТорг»

you have to give answers according this rules or explaining it. 

If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use 5 sentences maximum and keep the answer as concise as possible. 

{context}
Question: {question}
Helpful Answer:"""


# def get_rag_chain(vector_store):
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
#     rag_prompt_custom = PromptTemplate.from_template(prompt_template)
#
#     return (
#             {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
#             | rag_prompt_custom
#             | llm
#     )


def get_retriever_chain(vector_store):
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.1),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={
            "prompt": prompt,
        },
    )
