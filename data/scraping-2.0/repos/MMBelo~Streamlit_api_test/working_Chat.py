import os
import streamlit as st
import openai
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = st.secrets["key1"]
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],)

# Function to create or get the session state
def get_session_state():
    return st.session_state

#os.environ['OPENAI_API_KEY'] = st.secrets["key1"]

# App framework
st.title('🤖🍞  Talking toaster AI')

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

prompt = st.text_input('Ask the Toaster')

# Get or create the session state
state = get_session_state()

# Initialize conversation history if not present
if 'conversation_history' not in state:
    state.conversation_history = []

# Maintain conversation history
#conversation_history = []
# Prompt template
prompt_template = (
    "Your name is Talking Toaster. As an experienced Electric Engineer specializing in household appliances or electronic equipment, "
    "your task is to assist individuals with no technical background in identifying and addressing technical issues. Maintain a helpful, "
    "friendly, clear, and concise tone throughout. Start by briefly describing the product and confirming its equipment and model. "
    "Then, identify the issue and seek clarification with up to two simple, non-technical questions if needed. Provide a straightforward "
    "solution. Highlight common mispractices for the equipment. If the repair is too technical or potentially hazardous, advise seeking "
    "support from the equipment's brand or hiring a specialized technician. Answer: {topic}"
)
# Function to generate response using OpenAI API
def generate_response(prompt, conversation_history):
    if prompt:
        # Add current user prompt to the conversation history
        conversation_history.append(f"User: {prompt}")
        try:
            # Combine the conversation history with the prompt template
            combined_history = "\n".join(conversation_history)
            response = client.chat.completions.create(
            messages=[
                    {"role": "system", "content": "You are a helpful AI."},
                    {"role": "user", "content": combined_history + "\n" + prompt_template.format(topic=prompt)}
                ], model="gpt-3.5-turbo",
            )
            # Add AI response to the conversation history
            conversation_history.append(f"AI: {response.choices[0].message.content}") # (f"AI: {response['choices'][0]['message']['content']}")
            # Keep only the last 6 entries in the conversation history
            conversation_history = conversation_history[-6:]
            return response.choices[0].message.content #response['choices'][0]['message']['content']
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return None
# Display response
if st.button('Get Response'):
    response = generate_response(prompt, state.conversation_history)
    if response:
        st.text_area('Talking Toaster:', response, height=300)
# Display conversation history
st.text_area("Conversation History", "\n".join(state.conversation_history), height=300)

# Save the state so it persists across reruns
st.session_state = state
