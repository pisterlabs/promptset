
# app.py
from typing import List, Union

from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import streamlit as st
import requests
import json

PROMPT_TEMPLATE = """
Use the following pieces of context enclosed by triple backquotes to answer the question at the end.
\n\n
Context:
```
{context}
```
\n\n
Question: [][][][]{question}[][][][]
\n
Answer:"""

# with_rag=False

def init_page() -> None:
    st.set_page_config(
        page_title="XBC Bank AI Assistant"
    )
    st.header("XBC Bank AI Assistant")
    st.sidebar.title("Options")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in mardkown format.")
        ]
        st.session_state.costs = []


def select_llm() -> Union[ChatOpenAI, LlamaCpp]:
    #model_name="llama-2-7b-chat.Q5_K_M.gguf"
    model_name ="llama-2-7b-xbcfinetuned-q8_0-gguf"

    #model_name = st.sidebar.radio("Choose LLM:",  - Removded option button - Ahilan 12/12
    #                              ("gpt-3.5-turbo-0613", "gpt-4",
    #                               "llama-2-7b-chat.Q5_K_M.gguf"))

    
    # I am going to leave the temperature above mid range for better creativity - Removed slider - Ahilan 12/12
    temperature=0.0 # settting it to zero to get concerete answers with less maninupulation - Ahilan 12/17
    #temperature = st.sidebar.slider("Temperature:", min_value=0.0,
    #                                max_value=1.0, value=0.0, step=0.01)

    #No need to have the clause for chatGPT , leaaving it for future use - Ahilan 12/12
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=temperature, model_name=model_name)
    elif model_name.startswith("llama-2-"):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path=f"./models/{model_name}.bin",
            input={"temperature": temperature,
                   "max_length": 2000,
                   "top_p": 1
                   },
            callback_manager=callback_manager,
            verbose=False,  # True
        )


def get_answer(llm, messages) -> tuple[str, float]:
    if isinstance(llm, ChatOpenAI):
        with get_openai_callback() as cb:
            answer = llm(messages)
        return answer.content, cb.total_cost
    if isinstance(llm, LlamaCpp):
        return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0


def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

def extract_userquesion_part_only(content):
    """
    Function to extract only the user question part from the entire question
    content combining user question and pdf context.
    """
    content_split = content.split("[][][][]")
    if len(content_split) == 3:
        return content_split[1]
    return content

def process_llm_output(rest_response,llm_response) -> str:
   
    processed_output=''
    if((llm_response.upper().find('HOWEVER') > -1) or (llm_response.upper().find('JUST AN AI') > -1) or (llm_response.upper().find('PROPER AUTHORIZATION') > -1) or (llm_response.upper().find('CONSENT') > -1) or
     (llm_response.upper().find('ETHICAL') > -1) or (llm_response.upper().find('SECURITY') > -1) or (llm_response.upper().find('CAN YOU') > -1) or (llm_response.upper().find('PLEASE PROVIDE') > -1)): 
        processed_output=f'{rest_response} \n Hope I have fulfilled your request. Is there anything else I can do for you?'
    else:
        processed_output=f'{llm_response} \n Hope I have fulfilled your request. Is there anything else I can do for you?'
  
    return processed_output

def main() -> None:
    _ = load_dotenv(find_dotenv())

    init_page()
    llm = select_llm()
    init_messages()

    with_rag = st.sidebar.checkbox("## Use RAG ", False)
    # Supervise user input
    text_context=''
    if user_input := st.chat_input("Input your question!"):
        
        if(with_rag):
            #call account services
            accountservices_url = f'http://127.0.0.1:5000/processuserpmt?prompt={user_input}'
            json_result = requests.get(accountservices_url).json()
            print(json_result)
            text_context = json_result['result']
            user_input_w_context = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]) \
                .format(
                    context=text_context, question=user_input)
            st.session_state.messages.append(HumanMessage(content=user_input_w_context))

        else: 
            st.session_state.messages.append(HumanMessage(content=user_input))

        with st.spinner("AI Assistant is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        # Removed post processing hack with the fine tuned model - Ahilan 12/24
        #if(with_rag):
        #    revised_answer=process_llm_output(text_context,answer)
        #else:
        #    revised_answer=answer
        revised_answer=answer
        st.session_state.messages.append(AIMessage(content=revised_answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(extract_userquesion_part_only(message.content))
    # Removed cost display from sidebar leaving the cost calculation for future reference - Ahilan 12/12/23 
    #costs = st.session_state.get("costs", [])
    #st.sidebar.markdown("## Costs")
    #st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    #for cost in costs:
    #    st.sidebar.markdown(f"- ${cost:.5f}")


# streamlit run app.py
if __name__ == "__main__":
    main()