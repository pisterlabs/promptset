import os
import streamlit as st
from openai import OpenAI

client = OpenAI()

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets["key"]

# App framework
st.title('🤖🍞 Talking toaster AI')
prompt = st.text_input('Ask the Toaster')

# Maintain conversation history
conversation_history = []

# Prompt template
prompt_template = (
    "Your name is Talking Toaster, As an experienced Electric Engineer specializing in household appliances or electronic equipment, "
    "your task is to assist individuals with no technical background in identifying and addressing technical issues. Maintain a helpful, "
    "friendly, clear, and concise tone throughout. Start by briefly describing the product and confirming its equipment and model. "
    "Then, identify the issue and seek clarification with up to two simple, non-technical questions if needed. Provide a straightforward "
    "solution. Highlight common mispractices for the equipment. If the repair is too technical or potentially hazardous, advise seeking "
    "support from the equipment's brand or hiring a specialized technician. answer: {topic}"
)

# Function to generate response using OpenAI API
def generate_response(prompt, conversation_history):
    if prompt:
        # Add current user prompt to the conversation history
        conversation_history.append(f"User: {prompt}")

        try:
            # Combine the conversation history with the prompt template
            combined_history = "\n".join(conversation_history)
            response = client.completions.create(engine="GPT-3.5 Turbo",  # You can choose a different engine if needed
            prompt=combined_history + "\n" + prompt_template.format(topic=prompt),
            max_tokens=150)

            # Add AI response to the conversation history
            conversation_history.append(f"AI: {response.choices[0].text.strip()}")

            # Keep only the last 6 entries in the conversation history
            conversation_history = conversation_history[-6:]

            return response.choices[0].text.strip()

        except Exception as e:
            st.error(f"Error generating response: {e}")
            return None

# Display response
if st.button('Get Response'):
    response = generate_response(prompt, conversation_history)
    if response:
        st.text('Talking Toaster:', response)

# Display conversation history
st.text('\n'.join(conversation_history))










#import streamlit as st
#import os
#from langchain.chat_models import ChatOpenAI
#from langchain.llms import OpenAI
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain, SequentialChain
#from langchain.memory import ConversationBufferMemory
##from langchain.schema.messages import HumanMessage, SystemMessage
##from langchain import LangChain
#
#os.environ['OPENAI_API_KEY'] = st.secrets["key"]
#
## App framework
#st.title('🤖🍞 Talking toaster AI')
#
#prompt = st.text_input('Ask the Toaster')
#title_template = PromptTemplate(input_variables = ['topic'], template="Your name is Talking Toaster, As an experienced Electric \
#    Engineer specializing in household appliances or electronic equipment, your task is to assist individuals with no technical \
#
#        background in identifying and addressing technical issues. Maintain a helpful, friendly, clear, and concise tone throughout. \
#            Start by briefly describing the product and confirming its equipment and model. Then, identify the issue and seek \
#                clarification with up to two simple, non-technical questions if needed. Provide a straightforward solution. Highlight \
#                    common mis practices for the equipment. If the repair is too technical or potentially hazardous, advise seeking \
#                        support from the equipments brand or hiring a specialized technician. answer: {topic}")
## Memory
#title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
#
## Llms
#llm = OpenAI(temperature=0.9)
#
#
#title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
#
# Show stuff to the screen if there's a prompt

#
#def give_response(prompt):
#    if prompt:
#        #response = title_chain.run(topic=prompt)
#        response = 'I reached chatbotapp.py'
#
#        return response
#
#
#if __name__ == '__main__':
#    print(give_response(1))
