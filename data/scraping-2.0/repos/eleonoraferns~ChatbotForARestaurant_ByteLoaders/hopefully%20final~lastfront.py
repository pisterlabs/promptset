import streamlit as st
#import pandas as pd  # If not already imported
//from langchain_experimental.agents.agent_toolkits.csv.base import func1
from backend_code import func1:
from langchain.llms import OpenAI
def main():
    st.title("Chatbot Interface")
    
    # Create a text input field for user input
    user_input = st.text_input("User Input", "")

    # Define the chatbot agent and set its parameters
    agent = create_csv_agent(OpenAI(temperature=0), 'swiggy.csv')
    
    if st.button("Submit"):
        if user_input:
            # Get chatbot response
            response = agent.run(user_input)
            st.text("Chatbot Response:")
            st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
