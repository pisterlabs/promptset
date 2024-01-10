import openai
import re
import streamlit as st
from prompts import get_system_prompt
from prompts import checkSimilarity
from prompts import distilledbert_test, perform_sentiment_analysis
import pandas as pd
import matplotlib.pyplot as plt
import json

st.title("📺YOUTUBE BOT")

# Initialize the chat messages history
openai.api_key = st.secrets.OPENAI_API_KEY
if "messages" not in st.session_state:
    # system prompt includes table information, rules, and prompts the LLM to produce
    # a welcome message to the user.
    st.session_state.messages = [{"role": "system", "content": get_system_prompt()}]

# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

user_question=None
# display the existing chat messages
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        user_question=str(message["content"])
        st.write(message["content"])
        if "results" in message:
            st.dataframe(message["results"])

# If last message is not from assistant, we need to generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = ""
        resp_container = st.empty()
        for delta in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            response += delta.choices[0].delta.get("content", "")
            resp_container.markdown(response)

        message = {"role": "assistant", "content": response}
        # Parse the response for a SQL query and execute if available
        sql_match = re.search(r"```sql\n(.*)\n```", response, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1)
            conn = st.experimental_connection("snowpark")
            message["results"] = conn.query(sql)
            st.dataframe(message["results"])
            video_data = pd.DataFrame(message["results"])
            comments_list=[]
            if 'COMMENTS' in video_data.columns:
                #comments_list = video_data['COMMENTS'].tolist()
                num_records = len(video_data['COMMENTS'])
                for data in video_data['COMMENTS']:
                    comments_list.append(data)
                #comments_list = video_data['COMMENTS'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
                #1 for implementing analysis dynamically
                if user_question:
                    analysis_ques=checkSimilarity(user_question.lower())
                    # Check if the question is related to sentiment analysis
                    distilledbert_result=[]
                    if analysis_ques:
                        # Assume 'inputdata' is the relevant context for DistilledBERT  
                        if num_records != 0 :  
                            #distilledbert_result = distilledbert_test(user_question, " ".join(comments_list))
                            results = perform_sentiment_analysis(" ".join(comments_list))
                            #st.table(pd.DataFrame(results))
                            sentiment_counts = pd.DataFrame(results)['Sentiment'].value_counts()
                            total_comments = sentiment_counts.sum()
                            sentiment_percentage = (sentiment_counts / total_comments) * 100
                            fig, ax = plt.subplots()
                            ax.pie(sentiment_percentage, labels=sentiment_percentage.index, autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                            # Display the chart in Streamlit
                            st.pyplot(fig)
                            #st.bar_chart(sentiment_counts)
                            st.bar_chart(sentiment_percentage)
                        else:
                            distilledbert_result.append({"Answer":"No comments retrieved for further analysis"})
                        #st.write(f"{user_question} : {distilledbert_result}")
                        #st.dataframe(distilledbert_result)
                    else:
                        # Your existing logic to handle other types of questions
                        # ...
                        st.write("Oops, I haven't been trained to answer these questions yet. Here's a brief about the video content based on the comments")
                        if num_records != 0 :  
                            distilledbert_result = distilledbert_test("brief about content of the comments", " ".join(comments_list))
                        else:
                            distilledbert_result.append({"Answer":"No comments retrieved for further analysis"})
                        #st.write(f"{user_question} : {distilledbert_result}")
                        st.dataframe(distilledbert_result)
                #.1
        st.session_state.messages.append(message)
