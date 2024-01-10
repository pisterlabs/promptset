import streamlit as st
import re
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from cg_utils import *


def similarity_search(query:str, text:str) -> str:
    """Similarity search using LangChain, Bedrock's Titan embeddings and FAISS"""
    bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)
    sentences_list = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    faiss = FAISS.from_texts(sentences_list, bedrock_embeddings)
    results = faiss.similarity_search(query,1)
    result = ""
    for r in range(len(results)):
        result = result + "\n" + results[r].page_content
    if result:
        return result
    else:
        return "No matches for similarity search!"
    

def ask_fm(query:str, context:str) -> str:
    """Contextual content generation using LangChain, Bedrock and Anthropic Claude v2"""
    inference_parameters = {'max_tokens_to_sample':4096,
                            "temperature":0.1,
                            "top_k":5,
                            "top_p":0.2,
                            "stop_sequences": ["\n\nHuman"]
                            }
    fm = get_fm("anthropic.claude-v2")
    prompt_template = """
    Human: Use the following context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not tell the Human that you are using the context.

    {context}

    Question: {query}
    Assistant:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "query"]
    )
    result = fm(PROMPT.format(context=context, query=query))
    return result


def main():
    """Main function for text querying"""
    st.set_page_config(page_title="Text Query", layout="wide")
    css = '''
        <style>
            .stTextArea textarea {
                height: 190px;
                # color: #de8d0b;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                # padding-left: 5rem;
                # padding-right: 5rem;
            } 
            #divshell {
                border-top-right-radius: 7px;
                border-top-left-radius: 7px;
                border-bottom-right-radius: 7px;
                border-bottom-left-radius: 7px;
            }                         
        </style>
    '''
    st.write(css, unsafe_allow_html=True)
    st.header("Similarity Search Vs FM Contextual Query")
    st.write("Enter your query regarding the text and press Enter to see the results of a similarity search and a contextual query against an FM. You may modify the text or use your own text.")
    st.markdown("Refer the [Demo Overview](Solutions%20Overview) for a description of the solution.")
    default_text =  "London has cool, damp winters so a warm wool coat, hat, scarf, umbrella and gloves are good choices for Christmas there." + \
                    " Toronto has very cold, snowy winters so a heavy winter coat, sturdy boots, hat, scarf, and mittens are appropriate Christmas clothes in Toronto." + \
                    " Los Angeles has mild, sunny winter weather so light jackets, pants, and sweaters are appropriate Christmas attire in LA." + \
                    " New York has cold, often snowy winters so a warm winter coat, hat, gloves, and boots are wise Christmas clothing choices in New York." + \
                    " Melbourne has mild to warm summers during Christmas so light dresses, shorts, t-shirts, and sandals are suitable Christmas clothes in Melbourne."
    text = st.text_area('sentences',default_text, key="sentences_key",label_visibility="hidden",)
    search_string = st.text_input("Enter query:",key="search_string_key",label_visibility="visible")
    search_string_validation = st.empty()
    col1, col2 = st.columns([1,1])
    with col1:
        if search_string:
            st.markdown(f"<div id='divshell' style='background-color: #fbf1dc;'><p style='text-align: center;font-weight: bold;'>Similarity Search</p>{similarity_search(st.session_state.search_string_key,text)}<br /></div>", unsafe_allow_html=True)
    with col2:
        if search_string:
            st.markdown(f"<div id='divshell' style='background-color: #f1fdf1;'><p style='text-align: center;font-weight: bold;'>FM Contextual Query (anthropic.claude-v2)</p>{ask_fm(st.session_state.search_string_key,text)}<br /></div>", unsafe_allow_html=True)           

# Main   
if __name__ == "__main__":
    main()