import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import tempfile
import os


def main():

    load_dotenv()

    st.set_page_config(page_title="AskCSV")
    st.header("Ask your CSV 😪")

    user_csv = st.file_uploader("Upload your CSV file below", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Yo, ask me something from the CSV")

        llm = OpenAI(temperature=0)

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(user_csv.getvalue())
            tmp.close()
            tmp_path = tmp.name

        # Pass the file path to create_csv_agent
        agent = create_csv_agent(llm, tmp_path, verbose=True)

        # Delete the temporary file
        os.remove(tmp_path)

        if user_question is not None and user_question != "":
            response = agent.run(user_question)
            st.write(response)


if __name__ == "__main__":
    main()
