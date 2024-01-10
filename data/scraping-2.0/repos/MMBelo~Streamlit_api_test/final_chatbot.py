import os
import streamlit as st
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = st.secrets["key1"]
client = OpenAI()




#os.environ['OPENAI_API_KEY'] = st.secrets["key1"]

# App framework
st.title('🤖🍞  Talking toaster AI')

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

##### new code test#################################
##### new code test###################################### new code test#################################


# Get the product name after uploading the image
product_name = st.text_input('Enter the product name (e.g., Samsung1000)')

# Get the first query to start the conversation
first_query = st.text_input('What seems to be the issue?')

# Maintain conversation history
conversation_history = []
#
#if first_query:
#    # Get the second query from the user
#    second_query = st.text_input('Enter the second query')

# Initial prompt for the first user input
initial_prompt = (
    """Your name is Talking Toaster. As an experienced Electric Engineer specializing in household appliances or electronic equipment,
    your task is to assist individuals with no technical background in identifying and addressing technical issues. Maintain a helpful,
    friendly, clear, and concise tone throughout. Start by briefly describing the product and confirming its equipment and model.
    Then, identify the issue and seek clarification with up to two simple, non-technical questions if needed. Provide a straightforward
    solution. Highlight common mispractices for the equipment. If the repair is too technical or potentially hazardous, advise seeking
    support from the equipment's brand or hiring a specialized technician. User Question: {topic}"""
)

topic1 = """You are a funny old lady always mad about household appliance malfunctions,\
            acknowledge the product name and reply in a fun way {topic1}"""

# Function to generate response using OpenAI API
def generate_response(prompt, conversation_history, is_first_query):
    if prompt:
        # Add current user prompt to the conversation history
        conversation_history.append(f"User: {first_query}")
        try:
            # Combine the conversation history with the appropriate prompt based on whether it's the first query or not
            if is_first_query:
                combined_history = initial_prompt.format(topic=prompt)
                prompt_template = f'You just received a name of a product {product_name}.'

            else:
                combined_history = "\n".join(conversation_history) + "\n" + initial_prompt.format(topic=prompt)
                prompt_template = initial_prompt

            # Determine the temperature based on whether it's the first query or not
            temperature = 0.8 if is_first_query else 0.3

            # Make the OpenAI API call with the specified temperature
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI."},
                    {"role": "user", "content": combined_history}
                ],
                model="gpt-3.5-turbo",
                temperature=temperature,
            )

            # Add AI response to the conversation history
            conversation_history.append(f"AI: {response.choices[0].message.content}")

            # Keep only the last 6 entries in the conversation history
            conversation_history = conversation_history[-6:]

            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return None

# Display response for the first query
if st.button('Get First Response'):
    response = generate_response(first_query, conversation_history, is_first_query=True)
    if response:
        st.text_area('Talking Toaster:', response, height=300)

## Display response for the second query
#if st.button('Get Second Response'):
#    response = generate_response(second_query, conversation_history, is_first_query=False)
#    if response:
#        st.text_area('Talking Toaster:', response, height=300)

# Display conversation history
st.text_area("Conversation History", "\n".join(conversation_history), height=300)













##### new code test###################################### new code test#################################

####################
#if uploaded_image:
#        fridge = execute_1(uploaded_image)
#
#        a = lemmatizing_manual()
#
#        if fridge:
#
#            manual = file_selection(fridge)
#
#            st.write("I see that you have a "+ fridge + "please let me know how can i assist you?")
#
#            user_input = st.text_input('')
#
#            if user_input:
#
#                answer = retrieve_answer(user_input)
#
#                st.write(answer)
#

#########################

#
#first_prompt = st.text_input('Ask the Toaster')
#
## Maintain conversation history
#conversation_history = []
#
## first Response Prompt template # Add product
#first_prompt_template = ('You just recive a name of a product' + 'Samsung1000' + 'You are a funny old lady always mad about house apliences malfunctions, acknowledge the product name and reply in a fun way {topic1}')
#
## Function to generate response using OpenAI API
#def first_generated_response(first_prompt, first_conversation_history):
#    if first_prompt:
#        # Add current user prompt to the conversation history
#        first_conversation_history.append(f"User: {first_prompt}")
#        try:
#            # Combine the conversation history with the prompt template
#            combined_history = "\n".join(first_conversation_history)
#            first_response = client.chat.completions.create(
#            first_messages=[
#                    {"role": "system", "content": "acknowledge the product name and reply in a fun way."},
#                    {"role": "user", "content": combined_history + "\n" + first_prompt_template.format(topic1=first_prompt)}
#                ], model="gpt-3.5-turbo", temperature=0.9,
#            )
## Add AI response to the conversation history
#            first_conversation_history.append(f"AI: {first_response.choices[0].first_message.content}") # (f"AI: {response['choices'][0]['message']['content']}")
#
#            return first_response.choices[0].first_message.content #response['choices'][0]['message']['content']
#        except Exception as e:
#            st.error(f"Error generating response: {e}")
#            return None
## Display response
#    if st.button('Get Response'):
#        first_response = first_generated_response(first_prompt_template, first_conversation_history)
#        if first_response:
#            st.text_area('Talking Toaster:', first_response, height=300)
# Display conversation history
#st.text_area("Conversation History", "\n".join(conversation_history), height=300)


#
#
## Prompt template
#prompt_template = (
#    "Your name is Talking Toaster. As an experienced Electric Engineer specializing in household appliances or electronic equipment, "
#    "your task is to assist individuals with no technical background in identifying and addressing technical issues. Maintain a helpful, "
#    "friendly, clear, and concise tone throughout. Start by briefly describing the product and confirming its equipment and model. "
#    "Then, identify the issue and seek clarification with up to two simple, non-technical questions if needed. Provide a straightforward "
#    "solution. Highlight common mispractices for the equipment. If the repair is too technical or potentially hazardous, advise seeking "
#    "support from the equipment's brand or hiring a specialized technician. Answer: {topic}"
#)
## Function to generate response using OpenAI API
#def generate_response(prompt, conversation_history):
#    if prompt:
#        # Add current user prompt to the conversation history
#        conversation_history.append(f"User: {prompt}")
#        try:
#            # Combine the conversation history with the prompt template
#            combined_history = "\n".join(conversation_history)
#            response = client.chat.completions.create(
#            messages=[
#                    {"role": "system", "content": "You are a helpful AI. and you will be talking about" + 'Samsung1000'}, # get product
#                    {"role": "user", "content": combined_history + "\n" + prompt_template.format(topic=prompt)}
#                ], model="gpt-3.5-turbo", temperature=0.1
#            )
#
#
#            # Add AI response to the conversation history
#            conversation_history.append(f"AI: {response.choices[0].message.content}") # (f"AI: {response['choices'][0]['message']['content']}")
#            # Keep only the last 6 entries in the conversation history
#            conversation_history = conversation_history[-6:]
#            return response.choices[0].message.content #response['choices'][0]['message']['content']
#        except Exception as e:
#            st.error(f"Error generating response: {e}")
#            return None
## Display response
#    if st.button('Get Response'):
#        response = generate_response(prompt, conversation_history)
#        if response:
#            st.text_area('Talking Toaster:', response, height=300)
## Display conversation history
#st.text_area("Conversation History", "\n".join(conversation_history), height=300)
#
