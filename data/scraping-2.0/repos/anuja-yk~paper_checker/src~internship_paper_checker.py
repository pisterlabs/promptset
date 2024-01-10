# Objective = To create a sample UI using python and streamlit
import streamlit as st
import pandas as pd
import difflib
import openai
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

openai.api_key = os.environ["OPENAI_API_KEY"]


class PaperChecker:
    def __init__(self, data_filename):
        self.user_answer = ""
        self.data = []
        st.text("- Made by YHK")
        self.data_filename = data_filename

    # In build_ui, create page using Streamline.
    def build_ui(self):
        st.title("Paper Checking")

        if 'question' not in st.session_state:
            st.session_state.question = self.data[0]['Question']

        if 'user_answer' not in st.session_state:
            st.session_state.user_answer = ""

        if 'counter' not in st.session_state:
            st.session_state.counter = 0

        if 'num_records' not in st.session_state:
            st.session_state.num_records = self.num_records

        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = []

        st.text_area("Question is: ", key="question")

        self.user_answer = st.text_area("Write your answer in lower case here", max_chars=100, key='user_answer')
        if st.session_state.counter < st.session_state.num_records:
            submit_button = st.button(label='Submit', on_click=self.collect_responses_callback)
        else:
            score_button = st.button(label='View Score', on_click=self.compute_and_show_results_callback)

    def collect_responses_callback(self):
        st.session_state.user_answers.append(self.user_answer)
        st.session_state.counter = st.session_state.counter + 1

        if st.session_state.counter < st.session_state.num_records:
            st.session_state.question = self.data[st.session_state.counter]['Question']
            st.session_state.user_answer = ""

    def compute_and_show_results_callback(self):
        for data_record, session_record in zip(self.data, st.session_state.user_answers):
            data_record['User_Answer'] = session_record
        self.compute_scores()
        self.show_results()

    def read_data(self):
        self.df = pd.read_csv(self.data_filename)
        self.num_records = len(self.df)
        for i in range(self.num_records):
            row_dict = {'Question': self.df.loc[i, "Question"],
                        'Model_Answer': self.df.loc[i, "Answer"]}
            self.data.append(row_dict)

    # In compute_score, compile the score obtained by the user.
    def compute_scores(self):
        st.write("Computing score")
        for row in range(self.num_records):
            text1 = self.data[row]['Model_Answer']
            text2 = self.data[row]['User_Answer']
            score = self.compute_score(text1, text2)
            self.data[row]['Score'] = score

    def compute_score(self, text1, text2):
        # return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

        # resp = openai.Embedding.create(
        #     input=[text1.lower(), text2.lower()],
        #     engine="text-similarity-davinci-001")
        # embedding_a = resp['data'][0]['embedding']
        # embedding_b = resp['data'][1]['embedding']
        #
        # similarity_score = np.dot(embedding_a, embedding_b)
        # return similarity_score
        sentences = [text1.lower(), text2.lower()]

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Compute embedding for both lists
        embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
        embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

        return float(util.pytorch_cos_sim(embedding_1, embedding_2)[0][0])

    # In show_results, print results in table format.
    def show_results(self):
        st.write("Showing results", self.data)

    def run(self):
        self.read_data()
        self.build_ui()


if __name__ == "__main__":
    data_filename = "PaperCheckerTryout.csv"
    paper_checker = PaperChecker(data_filename)
    paper_checker.run()
