import os
import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# os.environ['OPENAI_API_KEY'] = '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'
# openai.api_key = os.getenv('OPENAI_API_KEY')


def gpt_classify_sentiment(prompt, emotions):
    system_prompt = f'''You are emotionally Intelligent assistant.
  Classify the sentiments of the users text with only one of the following emotions: {emotions}.
  After classifying the text , respond with the emotions only.'''
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=20,
        temperature=0
    )
    r = response['choices'][0].message.content
    if r == '':
        r = 'N/A'

    return r


# emotions = 'positive, negative'
# text = 'AI will take over the world.'
# print(gpt_classify_sentiment(text, emotions))
# os.environ['OPENAI_API_KEY'] = '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'

if __name__ == "__main__":
    openai.api_key = os.getenv('API_KEY')
    # openai.api_key = api_key

    col1, col2 = st.columns([0.85, 0.15])

    with col1:
        st.title("Zero Shot Sentiment Analysis")
    with col2:
        st.image('ai.png', width=70)

    with st.form(key='my_form'):
        default_emotions = 'positive, negative, neutral'
        emotions = st.text_input('Emotions', value=default_emotions)

        text = st.text_area(label='Text to classify: ')
        submit_button = st.form_submit_button(label='Check!')
        if submit_button:
            emotions = gpt_classify_sentiment(text, emotions)
            result = f"{text}=> {emotions} \n"
            st.write(result)
