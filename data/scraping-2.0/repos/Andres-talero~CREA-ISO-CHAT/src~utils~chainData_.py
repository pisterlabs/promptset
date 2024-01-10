from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.chains import ConversationalRetrievalChain, LLMChain, ConversationChain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from src.utils.vectorSearch import vector_search


memory = ConversationBufferWindowMemory(
    return_messages=True, k=6, input_key="human_input"
)


def load_chain():

    # System prompt message
    prompt_temp_system = PromptTemplate(template="""You are an expert ISO auditor, you work to Creasistemas company, your mission is only answer the user questions with the data information, don't answer question related with other topic different to ISO RULES, Creasistemas or the data information. Limit your responses to data information. In the answer include the source url of the information as citations in the end of the answer as a correct format link. 
    data:
    {context}
    
    """, input_variables=["context"],
    )

    system_template = SystemMessagePromptTemplate(prompt=prompt_temp_system)

    # User prompt message
    prompt_temp_human = PromptTemplate(template="{question}", input_variables=["question"],
                                       )

    human_template = HumanMessagePromptTemplate(prompt=prompt_temp_human)

    # ChatTemplate

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_template, MessagesPlaceholder(variable_name="history"), human_template])

    retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=5)

    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=AzureChatOpenAI(deployment_name="openai", temperature="0"),
    #     memory=memory,
    #     retriever=retriever,
    #     combine_docs_chain_kwargs={"prompt": chat_prompt},
    #     verbose=True,
    # )

    chain = LLMChain(
        llm=AzureChatOpenAI(deployment_name="openai", temperature="0"),
        memory=memory,
        prompt=chat_prompt,
        # return_source_documents=True,
        verbose=True
    )

    return chain


def get_response(question):
    data = vector_search(question)
    chain = load_chain()
    output = chain(
        {"context": data, "question": question, "human_input": question})
    print(output)
    # output = chain.run(question=question)
    return output
