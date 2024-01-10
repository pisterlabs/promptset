import argparse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

import custom_prompt
from constants import *

def pythia_chatbot(
        lake_name, 
        chain_type, 
        model_type, 
        retriever_distance_metric, 
        retriever_fetch_k, 
        retriever_mmr, 
        retriever_k, 
        supply_conversation_context
    ):
    if not lake_name:
        raise Exception("Must have non-null lake name")
    
    embeddings = OpenAIEmbeddings(disallowed_special=())

    db = DeepLake(dataset_path=f"hub://{username_activeloop}/{lake_name}", read_only=True, embedding_function=embeddings)

    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = retriever_distance_metric
    retriever.search_kwargs['fetch_k'] = retriever_fetch_k
    retriever.search_kwargs['maximal_marginal_relevance'] = retriever_mmr
    retriever.search_kwargs['k'] = retriever_k

    # import custom prompt
    qa_prompt = custom_prompt.CHAT_PROMPT
    if chain_type=="stuff" or chain_type=="map_rerank":
        prompt_arg_name = "prompt"
    elif chain_type=="map_reduce":
        prompt_arg_name = "combine_prompt"
    elif chain_type=="refine":
        prompt_arg_name = "refine_prompt"
    else:
        raise Exception("Invalid chain_type argument")

    # import chatbot
    model = ChatOpenAI(model_name=model_type) # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(model,chain_type=chain_type,retriever=retriever,combine_docs_chain_kwargs=dict([[prompt_arg_name, qa_prompt]]))

    chat_history = []

    while True:
        question = input("Ask a question: ")

        result = qa({"question": question, "chat_history": chat_history})
        if supply_conversation_context:
            chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run conversational retrieval chatbot based on indexed files in DeepLake VectorStore.')
    parser.add_argument("--lake_name", "-ln", type=str, help="Name of the lake you wish to draw from")
    parser.add_argument("--chain_type", "-ct", type=str, help="Name of the chain type to be used for the QA in the chatbot")
    parser.add_argument("--model_type", "-mt", type=str, help="Name of the LLM class to be used for the chatbot")
    parser.add_argument("--retriever_distance_metric", "-dm", type=str, help="Distance metric for retriever similarity function")
    parser.add_argument("--retriever_fetch_k", "-fk", type=int, help="Number of documents to fetch to pass to max_marginal_relevance algorithm")
    parser.add_argument("--retriever_k", "-k", type=int, help="Number of documents to return")
    parser.add_argument("--omit_conversation_context", "-occ", dest="supply_conversation_context", action="store_false", help="Flag for whether to omit feeding ongoing conversation as context into the QA chain")

    parser.set_defaults(model_type="gpt-3.5-turbo")
    parser.set_defaults(chain_type="stuff")
    parser.set_defaults(retriever_distance_metric="cos")
    parser.set_defaults(retriever_fetch_k=100)
    parser.set_defaults(retriever_k=10)
    parser.set_defaults(supply_conversation_context=True)

    args = parser.parse_args()

    pythia_chatbot(
        args.lake_name,
        args.chain_type,
        args.model_type, 
        args.retriever_distance_metric, 
        args.retriever_fetch_k, 
        True, 
        args.retriever_k,
        args.supply_conversation_context
    )