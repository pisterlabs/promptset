import streamlit as st
import openai
import tweepy

# Authenticate Twitter API
bearer_token = st.session_state.get('TWITTER_BEARER_TOKEN', '')
twitter_client = tweepy.Client(bearer_token)

# Authenticate OpenAI API
openai.api_key = st.session_state.get('YOUR_OPENAI_API_KEY', '')

def search_trends(topic, source):
    if source == "Twitter":
        trends = twitter_client.search_recent_tweets(topic, max_results=10)
        return [trend.text for trend in trends]
    elif source == "GPT":
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a trend generator. Provide the top trends based on the given topic. Give each trend in a new line. Don't repeat the same trend.Don't write introduction and end, only write the trends."},
                {"role": "user", "content": f"What are the top trends about {topic}?"}
            ]
        )
        
        # Here we assume that the model's response is a comma-separated list of trends.
        # You can adjust the response structure depending on how you train or instruct the model.
        trends = completion.choices[0].message["content"].split('\n')
        return trends

def get_trends(query, session_state):
    # Initialization
    if 'trend_engine' not in session_state:
        session_state['trend_engine'] = 'GPT'
    trend_function_choice = session_state['trend_engine']

    st.write(f"Searching {trend_function_choice} for ", query)

    # Add clear button
    if st.button("Clear"):
        st.session_state.get().clear()

    # Decision on which trend search function to use
    trends = search_trends(query, trend_function_choice)

    return trends
