import os
import json
import dotenv
import openai
import json
import pandas as pd
import streamlit as st

# .env file must have OPENAI_API_KEY and OPENAI_API_BASE
dotenv.load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
ENGINE = os.environ.get("ENGINE", "chat-gpt")
TEMPERATURE = 0.1
MAX_TOKENS = 200
TOP_P = 1
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

def SentimentAnalysis():
    """
    This function runs the openai.ChatCompletion.create function
    """
    system_msg = """
    You're an assistant for a customer support team.
    You're going to classify customer's feedback into 5 categories: positive, negative, neutral, question, and answer.
    Return classification results with given customer's feedback in JSON format.
    """

    examples_user_prompt ="""
    "I had an amazing experience at your restaurant! The food was delicious and the service was exceptional. Thank you so much!"
    "I was really disappointed with the product I received. It didn't meet my expectations and I feel like I wasted my money."
    "I don't have any strong feelings one way or the other about my recent purchase. It was just okay."
    "Can you tell me more about the warranty on this product?"
    """

    examples_assistant_res = """
    [{  
        "text": "I had an amazing experience at your restaurant! The food was delicious and the service was exceptional. Thank you so much!",  
        "category": "positive"  
    },  
    {  
        "text": "I was really disappointed with the product I received. It didn't meet my expectations and I feel like I wasted my money.",  
        "category": "negative"  
    },  
    {  
        "text": "I don't have any strong feelings one way or the other about my recent purchase. It was just okay.",  
        "category": "neutral"  
    },  
    {  
        "text": "Can you tell me more about the warranty on this product?",  
        "category": "question"  
    }]
    """
    
    st.markdown("# Customer Sentiment Review")
    st.markdown("This demo will show you how to use Azure OpenAI to classify customer's feedback into 5 categories: positive, negative, neutral, question, and answer.")
    with st.expander("Demo scenario"):
        st.image("https://github.com/hyssh/azure-openai-quickstart/blob/main/images/Architecture-demo-sentiments.png?raw=true")
        st.markdown("1. User will type text (multi lines) in the input box")
        st.markdown("2. __Web App__ sends the text to __Azure OpenAI__")
        st.markdown("3. __Azure OpenAI__ classify the text line by line and return the classification results in JSON format")
        st.markdown("4. __Web App__ shows the results on screen and user can download the results as CVS file")
    st.markdown("---")
    
    # sidebar for system messages and examples
    with st.sidebar:
        with st.container():
            st.info("Classify customer's feedback into 5 categories: positive, negative, neutral, question, and answer. Return classification results with given customer's feedback in JSON format.")
        sample_tab, system_tab = st.tabs(["samples", "system"])
        # samples
        with sample_tab:
            st.header("Examples")
            st.code("\"I like Azure.\"", language="html")
            st.code("\"I like Data Science.\"", language="html")

        with system_tab:
            st.header("System messages")
            st.code(system_msg, language="html")

    text_input_tab, csv_upload_tab = st.tabs(["Type customer feedback in text", "Upload CSV file with customer feedback"])

    with text_input_tab:
        with st.container():
            st.header("Type customer feedback in text")
            st.write("Use `double quote` for the each feedback and new line for additional feedback.")
            st.markdown("""
                        Example:
                        ```text
                        "I like Azure."
                        "I like Data Science."
                        ```
                        """)
            user_msg = st.text_area(label="Enter customer feedback here, add new line for each feedback")
            if st.button("Run sentiment review"):
                st.spinner("Running sentiment review...")
                results = pd.DataFrame(json.loads(run(user_msg, system_msg, examples_user_prompt, examples_assistant_res)), columns=["text", "category"])
                st.dataframe(results)

                # download results as csv file
                st.download_button(label="Download results as CSV file", data=results.to_csv(), file_name="results.csv", mime="text/csv")

        
    with csv_upload_tab:
        with st.container():
            st.header("Upload CSV file that has customer feedback")
            # upload csv file
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded_file is not None:
                st.spinner("Running sentiment review...")

                # read txt file
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = ["text"]
                with st.expander("View uploaded CSV file"):
                    st.dataframe(df)
                df_text_strings = "" 
                for _item in df["text"].tolist():
                    df_text_strings += f"{_item}\n"

                # run sentiment review
                # convert dataframe to string
                results = pd.DataFrame(json.loads(run(df_text_strings, system_msg, examples_user_prompt, examples_assistant_res)), columns=["text", "category"])

                st.dataframe(results)
                # download results as csv file
                st.download_button(label="Download results as CSV file", data=results.to_csv(), file_name="results.csv", mime="text/csv")
            else:
                st.empty()


# define custom function to run the openai.ChatCompletion.create function
def run(user_msg: str, system_msg: str, examples_user_prompt: str, examples_assistant_res: str):  
    """
    This function runs the openai.ChatCompletion.create function
    """
    messages = [{"role":"system", "content":system_msg}, 
                {"role":"user","content":examples_user_prompt},
                {"role":"assistant","content":examples_assistant_res},
                {"role":"user","content":user_msg}
                ]

    res = openai.ChatCompletion.create(
        engine=ENGINE,
        messages = messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        n=1
        )
    
    return res.choices[0].message['content']

if __name__ == "__main__":
    SentimentAnalysis()