import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import sys
from app_pages.backend.data_chain_backend import Chain
from PIL import Image

def create_page():
    st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>LLM Data Scientist Assistant with GPT</h2>", unsafe_allow_html=True)
    #st.markdown("<h5 style='text-align: center;padding-top: 0rem;'>by Sean and The Boys</h5>", unsafe_allow_html=True)

    chosen_dataset = None
    chosen_model = None
    prompt = None

    if "datasets" not in st.session_state:
        datasets = {}
        st.session_state["datasets"] = datasets

    else:
        # use the list already loaded
        datasets = st.session_state["datasets"]

    if "responses" not in st.session_state:
        responses = {}
        st.session_state["responses"] = responses

    else:
        # use the list already loaded
        responses = st.session_state["responses"]

    if "code_responses" not in st.session_state:
        code_responses = {}
        st.session_state["code_responses"] = code_responses

    else:
        # use the list already loaded
        code_responses = st.session_state["code_responses"]

    if "clicked_buttons" not in st.session_state:
        st.session_state["clicked_buttons"] = {}

    else:
        # use the list already loaded
        clicked_buttons = st.session_state["clicked_buttons"]


    if "chain" not in st.session_state:
        st.session_state["chain"] = Chain()

    else:
        # use the list already loaded
        chain = st.session_state["chain"]

    # create side bar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;padding-top: 0rem;'>Data Dashboard</h2>", unsafe_allow_html=True)
        data_container = st.empty()
        uploaded_file = st.file_uploader("", type=["CSV"])

        # radio buttons
        index_no = 0
        if uploaded_file:
            file_name = uploaded_file.name
            datasets[file_name] = pd.read_csv(uploaded_file)
            index_no = len(datasets) - 1
        # Radio buttons for dataset choice#     
        chosen_dataset = data_container.radio("Choose your data:", datasets.keys(), index=index_no)


    available_models = {"gpt-4-1106-preview":"gpt-4-1106-preview",
                        "gpt-4": "gpt-4",
                        "gpt-3.5-turbo": "gpt-3.5-turbo",
                        "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k"
                        }
    with st.sidebar:
        chosen_model = st.radio("Choose your model:", available_models.keys(), index = 1)
        #st.title(chosen_model)

    if bool(datasets):
        st.dataframe(datasets[chosen_dataset])
        prompt = st.chat_input("What can I help you with?")

        # first prompt pipeline
        if not responses:
            if prompt:
                #with st.chat_message("user"):
                    #st.markdown(prompt)
                response, code_response = chain.initial_chain(prompt, chosen_dataset, chosen_model)
        
                # append prompt and response
                responses[str(len(responses))] = prompt, response

                # append prompt and code response
                code_responses[str(len(code_responses))] = prompt, code_response

        # if this is not the first prompt 
        else:
            if prompt:
                with st.chat_message("user"):
                    st.markdown(prompt)
                response, code_response = chain.recreate(prompt, chosen_dataset, chosen_model)
                
                # append prompt and response
                responses[str(len(responses))] = prompt, response

                # append prompt and code response
                code_responses[str(len(code_responses))] = prompt, code_response

    # chat history
    printed_steps = set()  # empty set tracks duplicate

    if responses:
        for key, (user_message, assistant_responses) in responses.items():
            with st.chat_message("user"):
                st.markdown(user_message)

            # handle code response
            for response in assistant_responses:
                if response['role'] == 'assistant':
                    for i, (user_message, code_response) in code_responses.items():
                        for code in code_response:
                            if code["run_id"] == response["run_id"] and code["errors"] is None:
                                step = code["step_id"]
                                if step not in printed_steps:
                                    with st.chat_message("ai"):
                                        # if st.button("See code", key=code["step_id"]):
                                        st.code(code["input"])
                                        printed_steps.add(step)  # Add the input to the set

                    # then show assistant response
                    with st.chat_message("ai"):
                        st.markdown(response['value'])
                        if 'image_file_file_id' in response:
                            image_name = response['image_file_file_id']
                            image_path = f"images/{image_name}.png"

                            try:
                                image = Image.open(image_path)
                                st.image(image)

                                if image_name in clicked_buttons:
                                    with st.expander("See explanation"):
                                        st.write(clicked_buttons[image_name])
                                
                                elif st.button("Explain with Vision", key=image_name):
                                    try:
                                        vision_response = chain.vision(image_name)
                                        explanation = vision_response['choices'][0]['message']['content']

                                        with st.expander("See explanation"):
                                            st.write(explanation)

                                        clicked_buttons[image_name] = explanation
                                    except Exception as error:
                                        st.markdown("There was a vision error")
                                        st.markdown(error)       
                            except:
                                st.markdown("There was an error displaying the image")
