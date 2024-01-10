import streamlit as st
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("keyopenai") 

# Future-GPT role assignment with context and background knowledge
future_gpt_prompt = {
    "role": "system",
    "content": (
        "You are Future-GPT, a creative and innovative AI with the ability to foresee and simulate the future in a descriptive way. "
        "You have been informed about the current challenge: Develop an easy-to-use climate information and early warning system for farmers in South Africa. "
        "You are also aware of the conversation that the user had with a tree, where the tree shared its wisdom about the weather, its interconnectedness, patterns, and uncertainty. "
        "Your task is to imagine a future where the ideas from this conversation are considered and implemented more often, stronger than now or at all. "
        "Use hypothetical thinking, creative thinking, and make assumptions. Be very creative and use storytelling language to visualize the picture of how the future could look like. "
        "Consider the impacts on society, the environment, and the economy. "
        "Be creative and provide a detailed description of this potential future."
    )
}

st.title('A Call from the Future')
st.write('Rrrrring, rrrrring. Hello?')
st.write('...')
st.write('I am the future, i can assist you.')
st.write('Just tell me your ideas, after your conversation with the wise tree. Maybe we will find out, how the future could look like when your idea comes to life.')

# User inputs their idea
user_idea_content = st.text_input('Enter your idea here:')

# Initialize session_state if it doesn't exist
if 'future_gpt_response' not in st.session_state:
    st.session_state.future_gpt_response = ''

# Generate Future Scenario button
if st.button('Generate Future Scenario'):
    if user_idea_content:
        user_idea = {"role": "user", "content": user_idea_content}

        # Generate a response from Future-GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[future_gpt_prompt, user_idea],
        )

        st.session_state.future_gpt_response = response['choices'][0]['message']['content']
        st.write(st.session_state.future_gpt_response)
    else:
        st.write('Please enter your idea.')

# Summarize and Visualize button
if st.button('Summarize and Visualize'):
    if st.session_state.future_gpt_response:
        # Generate a summarization prompt
        summarization_prompt = {
            "role": "system",
            "content": "You are a helpful assistant for visualization prompts. Summarize the following scenario in two sentences: " + st.session_state.future_gpt_response
        }

        # Generate a response from GPT-3
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[summarization_prompt],
        )

        summary = response['choices'][0]['message']['content']
        st.write(summary)
        

        response = openai.Image.create(
              prompt=summary,
              n=1,
              size="1024x1024"
            )
        print(response)
        image_url = response['data'][0]['url']

   

        # Display the generated image
        st.image(image_url)
    else:
        st.write('Please generate a future scenario first.')
