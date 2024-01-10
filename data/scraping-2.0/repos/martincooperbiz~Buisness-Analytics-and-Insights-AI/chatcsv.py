import streamlit as st
import pandas as pd
from pandasai import PandasAI
from dotenv import load_dotenv
import os
from question_insight_generator import generate_questions_from_csv
import matplotlib

from pandasai.llm.starcoder import Starcoder
from pandasai.llm.falcon import Falcon
from pandasai.llm.openai import OpenAI

# Load your OpenAI API key from environment variables
def chat_with_csv(df, prompt):
    llm = Falcon(api_token="")
    llm = Starcoder(api_token="")
    llm = OpenAI(api_token="")
    
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df, prompt=prompt)
    return result

st.set_page_config(layout='wide')

st.title("Business Analytics Insight AI (BAIai)")

input_csv = st.file_uploader("Upload your document file", type=['csv'])


if input_csv is not None:
    st.info("Document Uploaded Successfully")
    
    try:
        data = pd.read_csv(input_csv)
        if not data.empty:
            st.dataframe(data, use_container_width=True)

            # Generate questions
            questions = generate_questions_from_csv(input_csv)

            st.text(questions)

            st.info("Chat Below")

            input_text = st.text_area("Enter your query", value=questions, height=200)

            if input_text is not None:
                if st.button("Ask AI"):
                    st.info("Your Query: " + input_text)

                    result = chat_with_csv(data, input_text)
                    if isinstance(result, pd.DataFrame):
                        st.write("Query Result:")
                        st.dataframe(result, use_container_width=True)
                    elif isinstance(result, str) and result.startswith("data:image/"):
                        st.write("Query Result (Image):")
                        st.image(result, use_container_width=True)

                        # Display the file path of the saved image
                        st.write(f"Image File Path: {result}")
                    else:
                        st.write("Query Result:")
                        st.text(result)
        else:
            st.error("The uploaded CSV file is empty.")
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
    except pd.errors.ParserError:
        st.error("Invalid CSV file format. Please upload a valid CSV file.")

# # Display an image with a custom caption
# st.image("exports\charts\temp_chart.png", caption="Custom Caption")

