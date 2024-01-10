import streamlit as st
import pandas as pd
import openai
import requests
from tqdm import tqdm
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]


## STREAMLIT APP

st.title('Headline Sentiment Analysis')
st.subheader("Upload your file in Excel format")
st.text("Format Example:")

headlines = {
    "Headline": ["Apple's new iPhone 12 is a big hit", "Apple is releasing Vision Pro in 2023", "Apple iPhone 15 Pro is the next major release"],
}
df_apple = pd.DataFrame(headlines)
st.dataframe(df_apple)

COMPANY_NAME = st.text_input("Enter Company Name", "Apple")
ROLE = st.text_input("Enter Role", 'Media Analyst')
PIN = st.text_input("Enter PIN", type="password")

## FUNCTIONS

COMPANY_INFO = f'''

    {COMPANY_NAME} is a Chinese multinational technology company.
    It operates primarily in the telecommunications equipment and consumer electronics industries.
    Its product lines include tablets, laptops, wearables, and other smart devices.

'''

system_prompt = f'''

    Forget all your previous instructions.
    Pretend you are a {ROLE} working for '{COMPANY_NAME}' analyzing news headlines related to {COMPANY_NAME}.
    You need to determine whether the news article is positive, negative, or neutral in terms of its impact on {COMPANY_NAME}.

    You should relate your news headlines analysis to the industry which {COMPANY_NAME} operates in.
    {COMPANY_INFO}

'''

user_prompt = f'''

    Analyze the news headlines and determine if the sentiment is: positive, negative or neutral for the company {COMPANY_NAME}.
    Return only a single word, either "POSITIVE", "NEGATIVE" or "NEUTRAL".
    Provide a clear and definitive answer towards positive or negative sentiment, minimizing neutral output.
    Then after a '||' seperator explain shortly why the sentiment is positive, negative, or neutral.
    Ensure that your analysis is accurate and based on factual information.

'''

def detect_sentiment_w_reason(text, temp = 1, p = 1):
    """ Prints what the sentiment of the text.

        Parameters
        ----------

        temp : int, optional
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic

        p : int, optional
            An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.

      """

    retries = 3
    sentiment = None

    while retries > 0:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + f"These are the headlines: {text}"}
        ]

        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages,
            # max_tokens = 3,
            # n = 1,
            stop = None,
            temperature = temp,
            top_p = p
        )

        response_text = completion.choices[0].message.content

        try:
            sentiment, reason = response_text.split(" || ")
            return sentiment, reason
        except:
            retries -= 1
            continue

    else:
        return "error", "error"

# ====================

st.divider()
st.subheader("Test Sentiment Analysis")

text = st.text_area("Enter Headline", "Apple's new iPhone 12 is a big hit")
if st.button("Analyze"):
    if PIN == st.secrets["PIN"]:
        sentiment, reason = detect_sentiment_w_reason(text)
        st.text("Sentiment: {}".format(sentiment))
        st.text_area("Reason", reason, disabled=True)
    else:
        st.error("Wrong PIN")


st.divider()

uploaded_file = st.file_uploader("Upload your headlines in Excel format here", type=['xlsx'])

if st.button("Analyze Headlines"):
    if PIN == st.secrets["PIN"]:
        if uploaded_file is None or COMPANY_NAME == "":
            if uploaded_file is None:
                st.error("Please upload an Excel file")
            if COMPANY_NAME == "":
                st.error("Please enter a company name")
            if ROLE == "":
                st.error("Please enter a role")

        else:
            try:
                df = pd.read_excel(uploaded_file)
                df = df.drop_duplicates(subset=['Headline'])

                st.text("Successfully uploaded headlines")
                st.text("Number of headlines: {}".format(len(df)))
                st.dataframe(df['Headline'].head(5))

                progress_text = "Analyzing Sentiment. Please wait. This may take a while..."
                my_bar = st.progress(0, text=progress_text)

            
                result = {
                    "Headline" : [],
                    "Sentiment" : [],
                    "Reason" : []
                }

                max_retries = 5
                wait_time = 30 # Time to wait before retrying failed request in seconds

                for i, row in tqdm(enumerate(df.iterrows()), total=len(df)):
                    retries = 0
                    while retries < max_retries:
                        try:
                            time.sleep(0.5)
                            sentiment, reason = detect_sentiment_w_reason(row[1]['Headline'])
                            result["Headline"].append(row[1]['Headline'])
                            result["Sentiment"].append(sentiment)
                            result["Reason"].append(reason)
                            my_bar.progress((i+1)/len(df), text=progress_text + f"  ({((i+1)/len(df))*100:.2f}%)")
                            # If the above code executed without any exception, break the while loop
                            break

                        except Exception as e:
                            st.error(f"Error occurred: {e}. Retrying...")
                            retries += 1
                            time.sleep(wait_time)

                    if retries == max_retries:
                        st.error(f"Failed to process the row after {max_retries} retries. Skipping...")
                        continue


                df_res_reason = pd.DataFrame(result)
                
                st.success('Sentiment Analysis Successful')

                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                csv = convert_df(df_res_reason)

                st.download_button(
                    label="Download Sentiment Result as CSV",
                    data=csv,
                    file_name='headline_sentiment.csv',
                    mime='text/csv',
                )
                
                st.subheader("Headline Sentiment Result")
                st.dataframe(df_res_reason)
                
            except Exception as e:
                st.error("Something went wrong... Try with fewer headlines or try again later. \n Error: {}".format(e))

    else:
        st.error("Wrong PIN")


