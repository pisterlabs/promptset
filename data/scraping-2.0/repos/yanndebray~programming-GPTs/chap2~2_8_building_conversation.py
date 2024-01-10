from openai import OpenAI
import streamlit as st

# Set the API key for the openai package
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=st.secrets['OPENAI_API_KEY'],
)

def chat(messages,model='gpt-3.5-turbo'):
  # Generate a response from the ChatGPT model
  completion = client.chat.completions.create(
        model=model,
        messages= messages,
  )
  response = completion.choices[0].message.content
  return response


# Initialization
if 'convo' not in st.session_state:
  st.session_state.convo = []

st.sidebar.title('ChatGPT-like bot 🤖')

# Display the response in the Streamlit app
for line in st.session_state.convo:
  st.chat_message(line['role']).write(line['content'])

# Create a text input widget in the Streamlit app
if prompt := st.chat_input():
  # Append the text input to the conversation
  with st.chat_message('user'):
    st.write(prompt)
  st.session_state.convo.append({'role': 'user', 'content': prompt })
  # Query the chatbot with the complete conversation
  with st.chat_message('assistant'):
    result = chat(st.session_state.convo)
    st.write(result)
  # Add response to the conversation
  st.session_state.convo.append({'role':'assistant', 'content':result})

# Debug
st.sidebar.write(st.session_state.convo)