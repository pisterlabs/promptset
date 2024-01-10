import openai
import streamlit as st


# Function to get GPT response
def get_gpt_response(api_key, model, role_description, message_history):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": role_description}] + message_history,
        temperature=0,
        max_tokens=2000
    )
    gpt_answer = response.choices[0]["message"]["content"]
    updated_message_history = message_history + [{"role": "assistant", "content": gpt_answer}]
    return gpt_answer, updated_message_history


# Streamlit app
st.title("Sanskrit AI")

# User input for API key
api_key = st.text_input("Enter your OpenAI API key:", type="password")
model = st.selectbox("Select your GPT model:", ["gpt-4 (recommended)", "gpt-3.5-turbo"])
if model == "gpt-4 (recommended)":
    model = "gpt-4"

# System role description
role_description = """    
You are an AI Sanskrit Expert specializing in teaching and interpreting Sanskrit texts, grammar, and literature.    
Your goal is to help students, researchers, and enthusiasts learn and understand Sanskrit language, texts, nuances, and concepts.    
You will provide guidance, suggestions, and answer relevant questions to ensure the best possible understanding of Sanskrit language and literature.    
You will always explain primarily in English, and use Sanskrit wherever required.  
"""

# Initialize message history if not already in session state
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# User input
placeholder_text = """यस्य कृत्यं न जानन्ति मन्त्रं वा मन्त्रितं परे।

कृतमेवास्य जानन्ति स वै पण्डित उच्यते ॥"""
user_prompt = st.text_area("What would you like to ask?:", height=150, placeholder=placeholder_text)

# Get GPT response
col1, col2, col3 = st.columns([1, 5, 1])
if col1.button("Submit", type="primary"):
    if len(str(api_key).strip()) == 0:
        st.error("Please enter your API key")
    else:
        # Add user input to message history
        st.session_state.message_history.append({"role": "user", "content": user_prompt})

        gpt_answer, updated_message_history = get_gpt_response(api_key, model, role_description,
                                                               st.session_state.message_history)
        st.write("Sanskrit AI:\n", gpt_answer)

        # Update message history
        st.session_state.message_history = updated_message_history

    # Clear message history
if col3.button("Reset", type="primary"):
    st.session_state.message_history = []

# Display conversation history
with st.expander("Conversation History"):
    for message in st.session_state.message_history:
        if message["role"] == "user":
            st.write(f"You: \n{message['content']}")
        else:
            st.write(f"Sanskrit AI: \n{message['content']}")
