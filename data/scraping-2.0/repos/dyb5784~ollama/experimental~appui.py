# app.py
# [!WARNING] Starting with version 0.1.79 the model format has changed from ggmlv3 to gguf. Old model files can be converted using the convert-llama-ggmlv3-to-gguf.py script in llama.cpp
# python3 vendor/llama.cpp/convert-llama-ggmlv3-to-gguf.py --input <path-to-ggml> --output <path-to-gguf>
# needed to downgrade pip install gguf==0.2.1s to get rid of AttributeError: 'TensorNameMap' object has no attribute 'get'running the conversion script
from typing import List, Union

from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st


def init_page() -> None:
    st.set_page_config(
        page_title="Local LLM GUI"
    )
    st.header("Local LLM UI")
    st.sidebar.title("Options")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="**[OMNICOMP]:COMPETENCE ACCESS STRATEGY! TEACHES MODEL TO THINK WELL ABOUT SKILLS:[OMNICOMP2.1R_v2]=>[OMNICMP2.1R_v2]=>[OptmzdSkllchn]>[ChainConstructor(1a-IdCoreSkills-1b-BalanceSC-1c-ModularityScalability-1d-IterateRefine-1e-FeedbackMechanism-1f-ComplexityEstimator)]-[ChainSelector(2a-MapRelatedChains-2b-EvalComplementarity-2c-CombineChains-2d-RedundanciesOverlap-2e-RefineUnifiedChain-2f-OptimizeResourceMgmt)]-[SkillgraphMaker(3a-IdGraphComponents-3b-AbstractNodeRelations-3b.1-GeneralSpecificClassifier(3b.1a-ContextAnalysis--3b.1b-DataExtraction--3b.1c-FeatureMapping--3b.1d-PatternRecognition--3b.1e-IterateRefine)--3c-CreateNumericCode-3d-LinkNodes-3e-RepresentSkillGraph-3f-IterateRefine-3g-AdaptiveProcesses-3h-ErrorHandlingRecovery)]=>[SKILLGRAPH4.1R_v2]**")
        ]
        st.session_state.costs = []


def select_llm() -> LlamaCpp:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("llama2uncensored-qquf",
                                   "noushermes-qquf"))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.7, step=0.01)
    #if model_name.startswith("gpt-"):
     #   return ChatOpenAI(temperature=temperature, model_name=model_name)
    #elif model_name.startswith("llama2-"):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    if {model_name}:
        print({model_name})
        print(temperature)
    return LlamaCpp(
            #model_path=f"C:\\Users\\danie\\.ollama\\models\\blobs\\sha256-4f8a6c6dfd38b66e4446f0cc938aad5db3b5e309447f8fb0ac7a359bb75a3fe6",
            #model_path=f"C://Users//danie//.ollama//models//blobs//sha256-71933c553b9c8c8720dc467b3788bb5625d4ab8b4b368c7c5b55f6fbae70931e",
            # nous-hermes sha256-ed1043d21e9811e0ba9e9d72f2c3b451cb63ffcc26032b8958cc486ddca005a4
            # model_path=f"C:\\Users\\danie\\.ollama\\models\\blobs\\sha256-ed1043d21e9811e0ba9e9d72f2c3b451cb63ffcc26032b8958cc486ddca005a4",
            model_path=f"C:\\Users\\danie\\.ollama\\models\\blobs\\{model_name}",
            
            callback_manager=callback_manager,
            temperature=temperature,
            max_tokens=4096,
            #top_p=1,  
            #rope_freq_scale=1, 
            #rope_freq_base= 1000,     
            #input={"temperature": temperature,
             #      "max_length": 2000,
              #     "top_p": 1
               #    },
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


def main() -> None:
    _ = load_dotenv(find_dotenv())

    init_page()
    llm = select_llm()
    init_messages()

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("LLM is thinking and typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


# streamlit run app.py
if __name__ == "__main__":
    main()