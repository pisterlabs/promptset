
import streamlit as st
import openai
import json
import pandas as pd

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """Welcome to the Movie Sentiment Analysis AI. 
            Your task is to analyze the sentiment of movie reviews and provide insights based on the sentiment expressed. 
            Please format your response in a JSON array containing the following details:
            - "movie_title" - the title of the movie being reviewed
            - "review_text" - the text of the movie review
            - "sentiment" - the sentiment category (positive, negative, neutral)
            - "comment" - any additional comments regarding the sentiment analysis
            Don't say anything at first. Wait for the user to say something.
        """    


st.title('Movie Reviews')
st.markdown('Input the Movie reviews that you want to know sentiment. \n\
            The AI will give you comments on how reviewer sentiments are.')

user_input = st.text_area("Enter some review text to generate sentiment:", "Your text here")


# submit button after text input
if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    # Show the response from the AI in a box
    st.markdown('**AI response:**')
    suggestion_dictionary = response.choices[0].message.content


    sd = json.loads(suggestion_dictionary)

    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
    st.table(suggestion_df)

