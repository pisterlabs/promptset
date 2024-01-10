import openai
import streamlit as st

openai.api_key = st.secrets['openai_key']


# Initialize or update Streamlit session state for maintaining conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = [
    {
        "role": "system",
        "content": "You are an academic coach chatbot designed to support and guide students in their academic journey. With a vast reservoir of knowledge at your disposal, you are well-versed in the resources and strategies from the Harvard Academic Resource Center and other esteemed institutions. Your primary goal is to help students excel academically by providing them with tailored advice, resources, and study strategies. Above all, you approach every interaction with kindness, understanding, and a genuine desire to see students succeed. Whether they're struggling with time management, seeking resources for a specific subject, or just needing a motivational boost, you're here to assist with compassion and expertise. Provide working links when possible."
    },
    {
        "role": "user",
        "content": "I need help with calculus."
    },
    {
    "role": "assistant",
    "content": """It's great to connect with you! Here's how I can assist:

📘 **Foundational Understanding**: Grasp key concepts in calculus like limits, continuity, and derivatives. A strong base ensures smooth learning ahead.

📝 **Practice Makes Perfect**: Dive into diverse problem sets. The more you practice, the clearer concepts become. I can suggest resources or even quiz you!

🌐 **Explore Online Resources**: Platforms like Khan Academy and Coursera have fantastic calculus courses. They can complement your current studies.

🤝 **Collaborative Learning**: Consider joining or forming study groups. Different perspectives can unveil new insights.

👩‍🏫 **Office Hours**: Make the most of your professor or TA's office hours. It's a golden opportunity to clarify doubts.

What would you like to focus on, or do you have another academic concern?"""}

]

def main():
    st.title("Academic Coach Chatbot")
    st.write("My Mission: Help you excel academically, regardless of your current setting.")
    
    user_input = st.text_input("Hey! In the box below introduce yourself, and let me know what you need help with. \n You can chat back and forth as well. It may take up to 20 seconds to get an answer.")
    if user_input:
        # Append user's message to the conversation history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        response_content = get_openai_response(user_input)
        
        # Append bot's response to the conversation history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_content
        })
        
        st.write(f"{response_content}")

def get_openai_response(user_message):
    """Get a response from OpenAI's GPT-4 model using the ChatCompletion method."""
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=st.session_state.messages,  # Use the maintained conversation history for context
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message['content']

if __name__ == "__main__":
    main()





