from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
import promptEngineer as prompt_engg

"""
Convert the vector database into a language model chain.

We use the `ChatVectorDBChain` provided by LangChain to convert
the vector database into a language model chain. We use the `OpenAI`
provided by LangChain to create the language model. The `temperature`
is used to control the randomness of the model. The `model_name` is
used to select the model to use. The `return_source_documents` is used
to return the source documents.

The function returns the language model chain.
"""
def vectordb_to_llmChain(vectordb):
    return ChatVectorDBChain.from_llm(OpenAI(
        temperature=0.2,
        model_name="gpt-3.5-turbo" 
    ), vectordb, return_source_documents=True)

"""
Takes the query from the client and returns the answer.

The function takes the query from the client and returns the answer
using the `query_parser` function. The `client_prompt` is the query
from the client. The `chat_history` is the history of the conversation
between the client and the server.

The function returns the answer.
"""
def query_parser(client_prompt, vectordb):
    qa = vectordb_to_llmChain(vectordb)
    sanitized_prompt, chat_history = prompt_engg.optimize_prompt(client_prompt)

    if sanitized_prompt['flag'] != "None" and sanitized_prompt['flag'] != "NONE" and sanitized_prompt['flag'] != "":
        return sanitized_prompt['help']

    result = qa({
        "question" : sanitized_prompt['optimized_prompt'],
        "chat_history" : chat_history 
    })
    source_documents = get_source_documents(result)
    final_answer = format_answer(result["answer"], list(set(source_documents)))
    return final_answer

"""
Get source documents from the result obtained.

The function takes the result obtained from the `query_parser` 
function and returns the list of source documents. The source
documents links are present in the `metadata` of the result's
`source_documents`. 
"""
def get_source_documents(result):
    return [
        result["source_documents"][i].metadata['source']
        for i in range(int(len(result["source_documents"])))
    ]

"""
Return formatted answer to the client.


"""
def format_answer(answer, source_documents):
    final_answer = str(answer) + "\n\n The source documents used were: \n"
    for doc in source_documents:
        final_answer += "\t- "
        final_answer += str(doc)
        final_answer += "\n"
    return final_answer