from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
from consts import CHROMA_SETTINGS, PERSIST_DIRECTORY

def main():
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(
        "when did okan founded?"
    )
    for doc in docs:
        print(doc.metadata["source"])
        print(doc.page_content)
    return
    kwargs = {
        "functions": [
            {
                "name": "sayHi",
                "description": "A function to greate the user when he write his name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The user name"
                        }
                    },
                    "required": ["name"]
                }
            }
        ],
    }
    llm = ChatOpenAI(
        temperature = 0,
        model_kwargs = kwargs
    )
    template = """
    Use the following pieces of context to answer the question at the end.
    You are a student assistant to help students apply to OKTamam System.
    Never say you are an AI model, always refer to yourself as a student assistant.
    If you do not know the answer say I will call my manager and get back to you.
    If the student wants to register you should ask him for some data one by one in separate questions:
    - Name
    - Phone
    - Email Address
    After the student enters all this data say Your data is saved and our team will call you.
    You must ask the student from his name first and then call sayHi function at once.


    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)

    memory = ConversationBufferMemory(
        input_key="question", memory_key="history", return_messages=True)

    conversation = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt, 
            "memory": memory,
        },
    )

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            # conversation.save(file_path="conversation.json")
            break
        # Get the answer from the chain
        response = conversation(query)
        print("final response")
        print(response)
        answer = response["result"]

        # Print the result
        print("\n> Answer:")
        print(answer)

if __name__ == '__main__':
    main()
