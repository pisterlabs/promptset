import streamlit as st

from genai import MessageTemplate, FunctionTemplate
from genai.eyfs import TextGenerator
from genai.streamlit_pages.utils import reset_state
import json
import os
import openai
from dotenv import load_dotenv
load_dotenv()

selected_model = "gpt-4-1106-preview"
temperature = 0.6

# Paths to prompts
PROMPT_PATH = "src/genai/sandbox/signals/data/"
PATH_SIGNALS_DATA = PROMPT_PATH + "signals_2023.json"
PATH_SYSTEM = PROMPT_PATH + "00_system.jsonl"
PATH_INTRO = PROMPT_PATH + "01_intro.jsonl"
PATH_ACTIONS = PROMPT_PATH + "intent_actions.json"

# Top signal function
path_func_top_signal = PROMPT_PATH + "func_top_signal.json"
path_prompt_top_signal = PROMPT_PATH + "prompt_top_signal.jsonl"
# Top three signals function
path_func_top_three_signals = PROMPT_PATH + "func_top_three_signals.json"
path_prompt_top_three_signals = PROMPT_PATH + "prompt_top_three_signals.jsonl"
# Intent detection function
path_func_intent = PROMPT_PATH + "func_intent.json"
path_prompt_intent = PROMPT_PATH + "prompt_intent.jsonl"
# Prompt: Impact on the user
path_prompt_impact = PROMPT_PATH + "02_signal_impact.jsonl"
# Prompt: Summary of different signals
path_prompt_choice = PROMPT_PATH + "03_signal_choice.jsonl"
# Prompt: Following up on user's question
path_prompt_following_up = PROMPT_PATH + "04_follow_up.jsonl"

def auth_openai() -> None:
    """Authenticate with OpenAI."""
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except Exception:
        openai.api_key = st.secrets["OPENAI_API_KEY"]


def read_jsonl(path: str) -> list:
    """Read a JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]
    

def generate_signals_texts(signals_data: dict, chosen_signals: list = None):
    signals = [signal["short_name"] for signal in signals_data]
    signals_titles = [signal["title"] for signal in signals_data]
    signals_summaries = [signal["summary"] for signal in signals_data]

    if chosen_signals is None:
        chosen_signals = signals

    # Combine titles and summaries into a single string
    signals_description = ""
    for short_name, title, summary in zip(signals, signals_titles, signals_summaries):
        if short_name in chosen_signals:
            signals_description += f"Signal '{short_name}': {title}\n{summary}\n\n"

    return signals_description    


def generate_action_texts(action_data: dict):
    actions = [a["name"] for a in action_data]
    action_descriptions = [a["description"] for a in action_data]
    action_text = ""
    for name, description in zip(actions, action_descriptions):
        action_text += f"Action '{name}': {description}\n\n"
    return action_text   

# Prepare the data
signals_data = json.load(open(PATH_SIGNALS_DATA, "r"))
signals_dict = {s['short_name']: s for s in signals_data}
signals_descriptions = generate_signals_texts(signals_data) 
signals = [s['short_name'] for s in signals_data]

actions_data = json.load(open(PATH_ACTIONS, "r"))
actions_descriptions = generate_action_texts(actions_data)
actions = [a['name'] for a in actions_data]


def predict_intent(user_message: str, messages: list) -> str:
    """Detect the intent of the user's message.
    
    Args:
        user_message (str): The user's message.
        messages (list): The history of messages.
    
    Returns:
        str: The intent of the user's message. Possible outputs are:
            - "explain": The user wants to know more about a signal.
            - "more_signals": The user wants to know more about a signal.
            - "follow_up": The user wants to know more about a signal.
            - "next_steps": The user wants to know more about a signal.
            - "none": The user's message does not match any intent.
    """
    func_intent = json.loads(open(path_func_intent).read())
    message_history = [MessageTemplate.load(m) for m in st.session_state.messages]
    message = MessageTemplate.load(path_prompt_intent)
    all_messages = message_history + [message]
    function = FunctionTemplate.load(func_intent)
    response = TextGenerator.generate(
            model=selected_model,
            temperature=temperature,
            messages=all_messages,
            message_kwargs={"intents": actions_descriptions, "user_input": user_message},
            stream=False,
            functions=[function.to_prompt()],
            function_call={"name": "predict_intent"},
    )    
    intent = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    return intent['prediction']


def predict_top_signal(user_message: str, signals: list) -> str:
    """Predict the top signal from the user's message.
    
    Args:
        user_message (str): The user's message.
    
    Returns:
        str: The top signal from the user's message.
    """
    # Function call
    func_top_signal = json.loads(open(path_func_top_signal).read())
    func_top_signal['parameters']['properties']['prediction']['enum'] = signals
    
    message = MessageTemplate.load(path_prompt_top_signal)
    function = FunctionTemplate.load(func_top_signal)

    response = TextGenerator.generate(
            model=selected_model,
            temperature=temperature,
            messages=[message],
            message_kwargs={"signals": signals_descriptions, "user_input": user_message},
            stream=False,
            functions=[function.to_prompt()],
            function_call={"name": "predict_top_signal"},
    )    
    top_signal = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    return top_signal['prediction']


def predict_top_three_signals(user_message: str, allowed_signals: list) -> list:
    """Predict the top signal from the user's message.
    
    Args:
        user_message (str): The user's message.
    
    Returns:
        str: The top signal from the user's message.
    """
    # Function call
    func_top_signals = json.loads(open(path_func_top_three_signals).read())
    func_top_signals['parameters']['properties']['prediction']['items']['enum'] = allowed_signals
    print(func_top_signals)
    message = MessageTemplate.load(path_prompt_top_three_signals)
    function_top_three = FunctionTemplate.load(func_top_signals)

    signals_descriptions_ = generate_signals_texts(signals_data, allowed_signals)

    response = TextGenerator.generate(
            model=selected_model,
            temperature=temperature,
            messages=[message],
            message_kwargs={"signals": signals_descriptions_, "user_input": user_message},
            stream=False,
            functions=[function_top_three.to_prompt()],
            function_call={"name": "predict_top_signals"},
    ) 
    top_signals = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    print(message)
    print(f"Prediction: {top_signals}")
    print(response)
    return top_signals['prediction']

def signals_bot(sidebar: bool = True) -> None:
    """Explain me a concept like I'm 3."""

    # Define your custom CSS
    # custom_css = """
    #     <style>
    #         /* Adjust the selector as needed */
    #         .stHeadingContainer {
    #             margin-top: -100px; /* Reduce the top margin */
    #         }
    #         #MainMenu {visibility: hidden;}
    #         footer {visibility: hidden;}
    #         header {
    #             visibility: hidden
    #         }
    #     </style>
    #     """

    # # Apply the custom CSS
    # st.markdown(custom_css, unsafe_allow_html=True)

    st.title("Signals chatbot")
    st.write("Let's discuss the future!")

    # First time running the app
    if "messages" not in st.session_state:
        # Record of messages to display on the app
        st.session_state.messages = []
        # Record of messages to send to the LLM
        st.session_state.history = []
        # Keep track of which state we're in
        st.session_state.state = "start" 
        # Fetch system and introduction messages
        st.session_state.signals = []
        
        # Add system message to the history
        system_message = read_jsonl(PATH_SYSTEM)[0]
        system_message = MessageTemplate.load(system_message)
        system_message.format_message(**{"signals": signals_descriptions})
        st.session_state.history.append(system_message.to_prompt())
        print(system_message.to_prompt())
        # Add the intro messages
        intro_messages = read_jsonl(PATH_INTRO)
        print(intro_messages)
        for m in intro_messages:
            st.session_state.messages.append(m)
            st.session_state.history.append(m)

    # Display chat messages on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user message
    user_message = st.chat_input("")
    if user_message:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.session_state.history.append({"role": "user", "content": user_message})
        
        if st.session_state.state == "start":
            intent = "new_signal"
            st.session_state.user_info = user_message
            st.session_state.state = "chatting"
        else:
            intent = predict_intent(user_message, st.session_state.history)
            print(intent)
            # intent = "following_up"

        if intent == "new_signal":
            # Predict the signal to explain
            allowed_signals = [s for s in signals if s not in st.session_state.signals]
            signal_to_explain = predict_top_signal(user_message, allowed_signals)
            st.session_state.signals.append(signal_to_explain)
            st.session_state.active_signal = signal_to_explain
            print(signal_to_explain)
            print(f"I have these signals in memory: {st.session_state.signals}")
            # Explain the signal
            instruction = MessageTemplate.load(path_prompt_impact)
            message_history = [MessageTemplate.load(m) for m in st.session_state.history]
            message_history += [instruction]
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""            
                for response in TextGenerator.generate(
                        model=selected_model,
                        temperature=temperature,
                        messages=message_history,
                        message_kwargs={
                            "signal": signals_dict[signal_to_explain]['full_text'], 
                            "user_input": st.session_state.user_info
                            },
                        stream=True,
                ):          
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")   
                message_placeholder.markdown(full_response)         
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.history.append({"role": "assistant", "content": full_response})

        elif intent == "more_signals":
            # Select the top 5 most relevant signals for the user
            # (remove the seen signals)
            # Provide an overview of the impacts of signal on the reader
            # Ask which one the bot should elaborate on
            allowed_signals = [s for s in signals if s not in st.session_state.signals]
            top_signals = predict_top_three_signals(st.session_state.user_info, allowed_signals)
            print(allowed_signals)
            print(top_signals)
            print(top_signals[0:3])
            # Explain the signal
            instruction = MessageTemplate.load(path_prompt_choice)
            top_signals_text = generate_signals_texts(signals_data, top_signals)
            message_history = [MessageTemplate.load(m) for m in st.session_state.history]
            message_history += [instruction]
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""            
                for response in TextGenerator.generate(
                        model=selected_model,
                        temperature=temperature,
                        messages=message_history,
                        message_kwargs={
                            "signals": top_signals_text, 
                            "user_input": st.session_state.user_info
                            },
                        stream=True,
                ):          
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")   
                message_placeholder.markdown(full_response)         
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.history.append({"role": "assistant", "content": full_response})

        elif intent == "following_up":
            print(st.session_state.active_signal)
            #Follow up the user's message
            instruction = MessageTemplate.load(path_prompt_following_up)
            message_history = [MessageTemplate.load(m) for m in st.session_state.history]
            message_history += [instruction]
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""            
                for response in TextGenerator.generate(
                        model=selected_model,
                        temperature=temperature,
                        messages=message_history,
                        message_kwargs={
                            "signal": signals_dict[st.session_state.active_signal]['full_text'], 
                            "user_input": user_message
                            },
                        stream=True,
                ):          
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")   
                message_placeholder.markdown(full_response)  

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.history.append({"role": "assistant", "content": full_response})

        # # Add user message to history
        # prompt = prompt2()
        # st.session_state.messages.append({"role": "user", "content": prompt.to_prompt()})
        # print(user_message)
        # # Generate AI response
        # with st.chat_message("assistant"):
        #     message_placeholder = st.empty()
        #     full_response = ""
        #     for response in TextGenerator.generate(
        #         model=selected_model,
        #         temperature=temperature,
        #         messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        #         message_kwargs= None,
        #         stream=True,
        #     ):
        #         full_response += response.choices[0].delta.get("content", "")
        #         message_placeholder.markdown(full_response + "▌")
        #     message_placeholder.markdown(full_response)   
        # # Add AI response to history
        # st.session_state.messages.append({"role": "assistant", "content": full_response})


def llm_call(
    selected_model: str, temperature: float, message: MessageTemplate, messages_placeholders: dict) -> str:
    """Call the LLM"""
    message_placeholder = st.empty()
    full_response = ""
    for response in TextGenerator.generate(
        model=selected_model,
        temperature=temperature,
        messages=[message],
        message_kwargs=messages_placeholders,
        stream=True,
    ):
        full_response += response.choices[0].delta.get("content", "")
        message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)

    return full_response


def prompt2():
    """
    Generate a prompt for an overview of the impact of signals on the user
    """
    prompt = MessageTemplate.load(data_path + "prompt2.json")
    return prompt

def main() -> None:
    """Run the app."""
    auth_openai()

    signals_bot(sidebar=False)


main()
