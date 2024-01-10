import os
import openai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate a response from the GPT-3.5 model
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
        n=1,
        stop=None,
        timeout=10,
    )
    return response.choices[0].text.strip()

# Streamlit app
def main():
    st.title("Chatbot with GPT-3.5")
    
    # Initial prompt
    initial_prompt = "How can I assist you today?"
    
    # Text Input option
    user_question = st.text_input(initial_prompt)
    
    if st.button("Submit"):
        if len(user_question) > 0:
            prompt = f"{initial_prompt}\n\nQ: {user_question}\nA:"
            response = generate_response(prompt)
            st.markdown(f"**Response:** {response}")
            
            # Follow-up question
            follow_up_question = st.text_input("Ask another question")
            
            if len(follow_up_question) > 0:
                prompt = f"Q: {follow_up_question}\nA: {response}\n\nQ:"
                response = generate_response(prompt)
                st.markdown(f"**Response:** {response}")

if __name__ == "__main__":
    main()
