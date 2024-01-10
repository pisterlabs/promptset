import streamlit as st
import os
from dotenv import load_dotenv
from icecream import ic
load_dotenv("../.env", override=True)


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate

from resources import get_es, get_es_semantic, load_openai_llm


#### Define the chat history
if "convo_history" not in st.session_state:
    st.session_state.convo_history =  []


################################################
# $$$$$$$\   $$$$$$\   $$$$$$\  
# $$  __$$\ $$  __$$\ $$  __$$\ 
# $$ |  $$ |$$ /  $$ |$$ /  \__|
# $$$$$$$  |$$$$$$$$ |$$ |$$$$\ 
# $$  __$$< $$  __$$ |$$ |\_$$ |
# $$ |  $$ |$$ |  $$ |$$ |  $$ |
# $$ |  $$ |$$ |  $$ |\$$$$$$  |
# \__|  \__|\__|  \__| \______/ 
################################################

#### Setup prompt templates and conversation objects
LLM=load_openai_llm()

TEMPLATE = """You are a helpful AI Chatbot that answers questions 
about past Presidential State of the Union addresses using only the following provided context.
If you can't answer the question using the following context say "I don't know".

Context: 
On {date}, President {administration} said the following:
{context}

Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["date", "administration", "context", "input"], 
    template=TEMPLATE
)
conversation = LLMChain(
    prompt=PROMPT,
    llm=LLM,
    verbose=True
)

def getSurroundingSourceParagraphs(result):
    index_name="elser_sotu_paragraphs"
    chunk_number = result.metadata["chunk"]
    sotu_id=result.metadata["sotu_id"]

    es = get_es()

    query = {
        "bool": {
            "must": [
                {"term": { "metadata.sotu_id.keyword": { "value": sotu_id}}},
                {
                    "range": {
                        "metadata.chunk": {
                            "gte": chunk_number-1,
                            "lte": chunk_number+1
                        }
                    }
                }
            ]
        }
    }

    sort = [
        {"metadata.chunk": {"order": "asc"}}
    ]
    
    results = es.search(index=index_name,query=query,size=3, sort=sort)
    return results


def get_ai_response(human_input):
    ## get langchain elasticsearch vector store
    es_elser = get_es_semantic("elser_sotu_paragraphs")
    
    ## do the elser vector search to get the best nist paragraph
    best_result = es_elser.similarity_search(human_input, k=1)[0]
    ic(best_result)
    
    ## use the metadata to retrieve surrounding 3 paragraphs
    results = getSurroundingSourceParagraphs(best_result)

    ## merge those paragraphs into a string
    context = []
    for result in results["hits"]["hits"]:
        context.append(result["_source"]["text"])
    context_string = "\n".join(context)

    url = best_result.metadata["url"]
    administration = best_result.metadata["administration"]
    date = best_result.metadata["date"]
    citation = f"<a href='{url}'>{administration} on {date}</a>"

    ## use the llm conversation to generate AI chat response
    ai_response = ic(conversation.run(
        input=human_input, 
        date=date,
        administration=administration,
        context=context_string)
    )
    
    ## return both the ai response and the semantic search result
    return ai_response, best_result, citation


## This is what to do when a new human input chat is received
def next_message(human_input):
    st.session_state.convo_history.append({"role":"user","message":human_input})

    ai_response, best_result, citation = get_ai_response(human_input=human_input)
    
    st.session_state.convo_history.append({"role":"ai","message":ai_response, "citation":citation})

################################################
# $$\                                                $$\     
# $$ |                                               $$ |    
# $$ |      $$$$$$\  $$\   $$\  $$$$$$\  $$\   $$\ $$$$$$\   
# $$ |      \____$$\ $$ |  $$ |$$  __$$\ $$ |  $$ |\_$$  _|  
# $$ |      $$$$$$$ |$$ |  $$ |$$ /  $$ |$$ |  $$ |  $$ |    
# $$ |     $$  __$$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |  $$ |$$\ 
# $$$$$$$$\\$$$$$$$ |\$$$$$$$ |\$$$$$$  |\$$$$$$  |  \$$$$  |
# \________|\_______| \____$$ | \______/  \______/    \____/ 
#                    $$\   $$ |                              
#                    \$$$$$$  |                              
#                     \______/                               
################################################


colHeaderA, colHeaderB = st.columns([6,1])
with colHeaderA:
    "# Ask a State of the Union Speech Question"
with colHeaderB:
    if st.button("Clear Memory"):
        st.session_state.convo_history = []


## this is the chat input at the bottom of the page
if human_input := st.chat_input("Ask a question ..."):
    next_message(human_input)

#### Loop through the conversation state and create chat messages
with st.container():
    if "convo_history" in st.session_state:
        for msg in st.session_state.convo_history:
            with st.chat_message(msg["role"]):
                st.write(msg["message"])
                if "citation" in msg:
                    citation = msg["citation"]
                    st.markdown(f"*Citation*: {citation}", unsafe_allow_html=True)

