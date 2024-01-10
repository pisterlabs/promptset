import os
from apikey import apikey
import datetime

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# Memory
conversation_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9)

# App framework
st.title('PolicyWeb ChatBot')

# Introduction & Initial Prompt for Conversational Interface
st.write("Hello! I'm here to understand your concerns. Please engage in a conversation about the issues you're facing.")

# LLM chain for chatbot responses
response_template = PromptTemplate(
    input_variables=['user_message'],
    template="{user_message}"
)
response_chain = LLMChain(llm=llm, prompt=response_template, verbose=True, output_key='response')

# Display chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.write(message)

# Get user input
user_message = st.text_input('Your message:')
if st.button("Send"):
    # Append user message to chat history
    st.session_state.chat_history.append(f"You: {user_message}")

    # Get chatbot response using LLM
    bot_response_data = response_chain.run(user_message)
    # bot_response = bot_response_data['response']
    bot_response = bot_response_data if isinstance(bot_response_data, str) else bot_response_data.get('response', '')

    st.session_state.chat_history.append(f"ChatBot: {bot_response}")

# End conversation and process user needs
if st.button("End Conversation"):
    full_conversation = " ".join(st.session_state.chat_history)
conversations = st.text_input('Your conversation with the chatbot:')


# create directory to save conversations
if not os.path.exists('conversations/'):
    os.makedirs('conversations/')


# # Prompt template for extracting concerns
# issue_extraction_template = PromptTemplate(
#     input_variables=['topic'],
#     template='Extract key issues related to outcomes that can be addressed by governmental policy from the following conversation: {topic}'
# )
# Prompt template for extracting concerns
issue_extraction_template = PromptTemplate(
    input_variables=['topic'],
    template='Given the following conversation, extract general concerns that emphasize personal effects or outcomes without referencing specific policies or solutions. For example, if someone mentions "I wish there were more bike lanes for safety", the extracted concern should be "I am concerned about the accessibility and safety of biking in my area". Conversation: {topic}'
)

# Chain for issue extraction
issue_extraction_chain = LLMChain(llm=llm, prompt=issue_extraction_template, verbose=True, output_key='concerns', memory=conversation_memory)

# Q Methodology Configuration
q_distribution = [-3, -2, -2, -1, -1, -1, 0, 0, 1, 1, 1, 2, 2, 3]

# ... [rest of the imports and initializations]

# Function to save concerns to a text file
def save_concerns_to_file(concerns):
    with open('/mnt/data/concerns.txt', 'w') as file:
        for concern in concerns:
            file.write(f"{concern}\n")


# Modified function to save concerns and their rankings to a text file
def save_concerns_and_rankings_to_file(filename, concerns, rankings):
    with open(filename, 'w') as file:
        for concern in concerns:
            file.write(f"{concern} - Ranking: {rankings[concern]}\n")

def structure_user_needs(concerns, rankings):
    """Convert extracted concerns into a structured format."""
    structured_needs = []
    for idx, concern in enumerate(concerns):
        concern_type = determine_concern_type(concern)
        need = {
            "concern_id": idx + 1,
            "description": concern,
            "type": concern_type,
            "urgency": "High" if rankings[concern] > 1 else "Medium" if rankings[concern] == 0 else "Low"
        }
        structured_needs.append(need)
    return structured_needs

# Prompt template for concern type determination
type_determination_template = PromptTemplate(
    input_variables=['concern'],
    template="Classify the following concern into a category such as Infrastructure, Healthcare, Economy, etc.: {concern}"
)

# Chain for concern type determination
type_determination_chain = LLMChain(llm=llm, prompt=type_determination_template, verbose=True, output_key='type')

def determine_concern_type(concern):
    """Use LLMChain to determine the type of a given concern."""
    response = type_determination_chain.run(concern)
    return response  # Directly return the string response


def generate_policy_cards(user_needs):
    """Simulate LLM function call to generate policy cards. replace and Use the GPT function call here"""
    policy_cards = []
    for need in user_needs:
        card = {
            "policy_id": f"policy_{need['concern_id']}",
            "title": f"Policy addressing {need['description']}",
            "description": f"Solution to address the concern of {need['description']}",
            "impact": "Expected positive impact",
            "cost": "Estimated cost",
            "timeframe": "Short-term"
        }
        policy_cards.append(card)
    return policy_cards



if conversations:
    # Extract concerns only once
    if 'concerns' not in st.session_state:
        # Split concerns by full stops
        st.session_state.concerns = [c.strip() for c in issue_extraction_chain.run(conversations).split('.') if c]
        st.session_state.current_concern_index = 0
        st.session_state.q_rankings = {}
        # Generate a unique file name based on the current timestamp
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st.session_state.filename = f'conversations/conversation_{timestamp_str}.txt'


    # If not all concerns are ranked, display the next one
    if st.session_state.current_concern_index < len(st.session_state.concerns):
        concern = st.session_state.concerns[st.session_state.current_concern_index]
        st.subheader(f"Rank the concern: {concern}")
        st.write("Please rank the concern based on its importance to you for policy creation.")

        # Show slider for current concern
        ranking = st.slider(concern, min(q_distribution), max(q_distribution), format="%d")
        
        # Save ranking to session state
        st.session_state.q_rankings[concern] = ranking

        # Next concern button
        if st.button("Next"):
            st.session_state.current_concern_index += 1
    else:
        st.write("Thank you for ranking all concerns!")
        st.write(st.session_state.q_rankings)

        # Structure the concerns
        structured_needs = structure_user_needs(st.session_state.concerns, st.session_state.q_rankings)
        print(structured_needs)
        # # Generate policy cards Awaleh you can use your function here 
        # policy_cards = generate_policy_cards(structured_needs)

        # # Display policy cards (for demonstration)
        # for card in policy_cards:
        #     st.write(card)



        # Save concerns and rankings to file when all concerns have been ranked
        save_concerns_and_rankings_to_file(st.session_state.filename, st.session_state.concerns, st.session_state.q_rankings)
        # Save concerns and rankings to file when all concerns have been ranked
        # save_concerns_and_rankings_to_file(st.session_state.concerns, st.session_state.q_rankings)


    with st.expander('Conversation History'):
        st.info(conversation_memory.buffer)
