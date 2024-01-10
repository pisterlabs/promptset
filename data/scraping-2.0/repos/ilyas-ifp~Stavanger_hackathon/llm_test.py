from dotenv import load_dotenv

import os
from langchain.chains import RetrievalQA
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationSummaryBufferMemory,ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from rag import *
from sklearn.metrics.pairwise import cosine_similarity
from graph_creation import create_graph

load_dotenv(".env.shared")
load_dotenv(".env.secret")


openai_client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)

llm = AzureChatOpenAI(
    deployment_name="gpt4",
    model_name="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
db = FAISS.load_local("data/embeddings",openai_client.embeddings.create(model=os.environ["ADA002_DEPLOYMENT"],input=""))
dict = db.docstore._dict


def extract_emb_id(graph):
    emb_id_map = {}
    for node, attr in graph.nodes(data=True):
        emb = attr.get('attribute')  # Replace 'emb' with the actual key for embeddings
        id = attr.get('id')  # Replace 'id' with the actual key for IDs

        if emb is not None and id is not None:
            emb_id_map[node] = {'emb': emb, 'id': id}

    return emb_id_map

def cosine_similarity_(vec_a, vec_b):
    """Compute the cosine similarity between two vectors."""
    return 1 - cosine_similarity(vec_a, vec_b)

def get_neighbors(G, node, num_neighbors=3):
    """ Get up to num_neighbors neighbors of a given node """
    if isinstance(node,str) :
        neighbors = list(G.neighbors(node))
    else :
        neighbors = list(G.neighbors(node[0]))  
    return neighbors[:num_neighbors]

def get_neighbors_of_neighbors(G, node, num_neighbors=2):
    """ Get unique neighbors for each neighbor of the given node """
    neighbors = get_neighbors(G, node, num_neighbors)
    unique_neighbors_of_neighbors = set()
    for neighbor in neighbors:
        for n in get_neighbors(G, neighbor, num_neighbors):
            unique_neighbors_of_neighbors.add(n)
    # Optionally, remove the original node if it appears in the set
    # unique_neighbors_of_neighbors.discard(node)
    return unique_neighbors_of_neighbors

def find_most_similar_node(emb_id_map, query_embedding):
    max_similarity = -1
    most_similar_node = None

    for node, data in emb_id_map.items():
        node_embedding = data['emb']
        similarity = cosine_similarity_(query_embedding, node_embedding)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_node = node

    return most_similar_node, max_similarity

def retrieval_function(graph,query,emb_id_map) :
    context = []
    query = compute_embeddings(query)
    node = find_most_similar_node(emb_id_map,query)
    value = graph.nodes[node[0]].get('id')
    text = dict[value].page_content
    context.append(text)
    neighbors = get_neighbors_of_neighbors(graph,node,num_neighbors=2)
    for n in neighbors :
        value = graph.nodes[n].get('id')
        text = dict[value].page_content
        context.append(text)
    print(len(context))
    return context    

    
     
def create_chain() :
    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """You are a nice chatbot having a conversation with a human. The user_input will be as follow :
                'question :' user question and some context to help you answering the question.
                You must respond the question using the context
                """
            ),
            # The `variable_name` here is what must align with memory
            # MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    

    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=None)
    return conversation

def main():
    # q = Queue()
    # callback_fn = StreamingStdOutCallbackHandlerYield(q)
    llm  = create_chain()
    # output_variable = agent_chain
    while True:
            user_input = input("Enter something (type 'exit' to quit): ")
            if user_input == 'exit':
                print("Exiting the program.")
                break
            else :
                return llm({"question": user_input})
            
if __name__ == "__main__":
    llm  = create_chain()
    G = create_graph()
    emb_id_map = extract_emb_id(G)
    # output_variable = agent_chain
    while True:
            user_input = input("Enter something (type 'exit' to quit): ")
            if user_input == 'exit':
                print("Exiting the program.")
                break
            else :
                context = retrieval_function(G,user_input,emb_id_map)
                context =  ', '.join(context)
                user_input = 'question :' + user_input + 'here is some context that you may use :' + context
                print( llm({"question": user_input})['text'])  