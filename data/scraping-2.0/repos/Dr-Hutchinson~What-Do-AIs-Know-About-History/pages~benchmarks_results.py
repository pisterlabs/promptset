import os
import openai
import streamlit as st
from datetime import datetime as dt
import pandas as pd
from numpy import mean
import pygsheets
from re import search
import time
from google.oauth2 import service_account

def app():

    #st.set_page_config(layout="wide")

    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)

    gc = pygsheets.authorize(custom_credentials=credentials)

    euro_sheet = gc.open('high_school_european_history_test')
    ap_euro = euro_sheet.sheet1
    us_sheet = gc.open('high_school_us_history_test')
    ap_us = us_sheet.sheet1
    world_sheet = gc.open('high_school_world_history_test')
    ap_world = world_sheet.sheet1
    benchmarks_sheets = gc.open('benchmark_tests')
    benchmarks = benchmarks_sheets.sheet1

    df1 = pd.DataFrame(ap_euro, index=None)
    df2 = pd.DataFrame(ap_us, index=None)
    df3 = pd.DataFrame(ap_world, index=None)

    df4_preheaders = pd.DataFrame(benchmarks, index=None)
    df4 = df4_preheaders.rename(columns=df4_preheaders.iloc[0]).drop(df4_preheaders.index[0])

    euro_random = df1.sample()
    us_random = df2.sample()
    world_random = df3.sample()

    st.header('What Does an AI "Know" About History?')

    st.subheader("Instructions:")
    st.write("This app permits you to test your historical knowledge against GPT-3. Choose one of the A.P. categories on the left to load a random question. Select an answer in the Human Interface to see if you answered correctly. You can then compare your answer against the AI in the GPT-3 Interface.")

    st.subheader("Implications:")
    st.write("Recent advances in machine learning have resulted in the creation of AI models capable of accurate recall of historical knowledge. These abilities have been measured by testing AIs on a commonly-assessed form of historical knowledge: the Advanced Placement (A.P.) curriculums for U.S., European, and World History. The first major assessment came from a [January 2021 study](https://arxiv.org/pdf/2008.02275.pdf) by machine learning researchers led by [Dan Hendryks](https://people.eecs.berkeley.edu/~hendrycks/), with GPT-3 initially achieving over 50% accuracy in these fields. By January 2022 a new model of GPT-3 demonstrated significant advances in performance, approaching near expert-level accuracy (80%) in two historical fields. You can see the more information and the full results from the most recent replication of the Hendryks study [here](https://github.com/Dr-Hutchinson/gpt-3_history_benchmark_results).")


    with st.sidebar.form(key ='Form2'):
        field_choice = st.radio("Choose the subject:", ["U.S. History", "European History", "World History"])
        def delete_sessions():
            for key in st.session_state.keys():
                del st.session_state[key]

        button2 = st.form_submit_button("Click here to load another question")

        if button2:
            delete_sessions()
            st.experimental_rerun()

    if field_choice == "U.S. History":
        field = us_random
    elif field_choice == "European History":
        field = euro_random
    else:
        field = world_random

    question_number1 = str(field.index[0])
    question = field.iloc[0][0]
    option_a = field.iloc[0][1]
    option_b = field.iloc[0][2]
    option_c = field.iloc[0][3]
    option_d = field.iloc[0][4]
    answer = field.iloc[0][5]

    benchmarks_question_number = df4.loc[df4['question_number'] == question_number1]
    question_check = not benchmarks_question_number.empty
    is_question_already_in_benchmarks = str(question_check)


    if answer == "A":
        answer_response = option_a
    elif answer == "B":
        answer_response = option_b
    elif answer == "C":
        answer_response = option_c
    else:
        answer_response = option_d

    if 'field' not in st.session_state:
        st.session_state.field = field_choice

    if 'question_number1' not in st.session_state:
        st.session_state.question_number1 = question_number1

    if 'question' not in st.session_state:
        st.session_state.question = question

    if 'option_a' not in st.session_state:
        st.session_state.option_a = option_a

    if 'option_b' not in st.session_state:
        st.session_state.option_b = option_b

    if 'option_c' not in st.session_state:
        st.session_state.option_c = option_c

    if 'option_d' not in st.session_state:
        st.session_state.option_d = option_d

    if 'answer' not in st.session_state:
        st.session_state.answer = answer

    if 'answer_response' not in st.session_state:
        st.session_state.answer_response = answer_response

    col1, col2 = st.columns([1,1])

    with col1:
        with st.form('form1'):
            st.subheader("Human Interface:")

            st.write('Read the question below, and select the best answer. Click on the "Submit Your Answer" button to see if you answered correctly.')

            st.write("Question #" + st.session_state.question_number1 + ":" + "\n\n" + st.session_state.question)
            submit_answer = st.radio("Choose from the following options:", ["A: " + st.session_state.option_a, "B: " + st.session_state.option_b, "C: " + st.session_state.option_c, "D: " + st.session_state.option_d])
            button1 = st.form_submit_button("Submit your Answer")

            if button1:

                fullstring = st.session_state.answer + ": " + st.session_state.answer_response
                substring = submit_answer

                if substring in fullstring:
                    st.write("Correct")
                else:
                    st.write("Incorrect")

                st.write("Answer - " + st.session_state.answer + ": " + st.session_state.answer_response)

    with col2:
        with st.form('form3'):
            st.subheader("GPT-3 Interface")
            st.write("Click on the button below to pose the question to GPT-3")
            button3 = st.form_submit_button("Submit Question")

            if button3:
                os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
                openai.api_key = os.getenv("OPENAI_API_KEY")

                summon = openai.Completion.create(
                        model='text-davinci-002',
                        prompt=st.session_state.question +  "A: " + st.session_state.option_a +  "B: " + st.session_state.option_b +  "C: " + st.session_state.option_c + "D: " + st.session_state.option_d,
                        temperature=0,
                        max_tokens=50)

                response_json = len(summon["choices"])

                for item in range(response_json):
                    output = summon['choices'][item]['text']

                output_cleaned = output.replace("\n\n", "")

                if 'output' not in st.session_state:
                    st.session_state.output = output_cleaned

                fullstring = st.session_state.answer + ": " + st.session_state.answer_response
                substring = st.session_state.output

                if substring in fullstring:
                    correct_status = 'correct'
                    st.write("GPT-3's Response: Correct")
                else:
                    correct_status = 'incorrect'
                    st.write("GPT-3's Response: Incorrect")

                st.write(st.session_state.output)

                def ranking_collection():
                    now = dt.now()
                    sh4 = gc.open('benchmark_tests')
                    wks4 = sh4[0]
                    cells = wks4.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
                    end_row = len(cells)
                    end_row_st = str(end_row+1)
                    d4 = {'field':[st.session_state.field], 'question_number':[st.session_state.question_number1],'correct_answer':[st.session_state.answer + ": " + st.session_state.answer_response], 'output_answer':[st.session_state.output], 'correct_status':[correct_status], 'time':[now]}
                    df4 = pd.DataFrame(data=d4, index=None)
                    wks4.set_dataframe(df4,(end_row+1,1), copy_head=False, extend=True)

                ranking_collection()


        def us_history_data():
            sh1 = gc.open('us_history_benchmark_results')
            sh2 = gc.open('benchmark_tests')
            wks1 = sh1[0]
            wks2 = sh2[0]
            now = dt.now()
            data = wks1.get_as_df(has_header=True, index_col=None)
            data2 = wks2.get_as_df(has_header=True, index_col=None)
            data['time'] = pd.to_datetime(data['time'])
            mask = (data['time'] > '5/11/2022 11:20:00') & (data['time'] <= now)
            data = data.loc[mask]
            field_value = "U.S. History"
            total_attempts = data["correct_status"].count()

            field_data = data[data['field'] == 'U.S. History']

            correct_data = field_data[field_data['correct_status'] == 'correct']

            incorrect_data = field_data[field_data['correct_status'] == 'incorrect']

            st.write('In May 2022, GPT-3 correctly answered {} out of {} U.S. History questions, for a {:.2f}% accuracy rate. Click [here](https://github.com/Dr-Hutchinson/gpt-3_history_benchmark_results/blob/main/us_history_benchmark_results.csv) to see the results.'.format(len(correct_data), len(field_data), len(correct_data)/len(field_data)*100))
            st.write("Below is GPT-3's accuracy on all questions to date.")
            st.bar_chart(data2['correct_status'].value_counts())

        def euro_history_data():
            sh1 = gc.open('european_history_benchmark_results')
            sh2 = gc.open('benchmark_tests')
            wks1 = sh1[0]
            wks2 = sh2[0]
            now = dt.now()
            data = wks1.get_as_df(has_header=True, index_col=None)
            data2 = wks2.get_as_df(has_header=True, index_col=None)
            data['time'] = pd.to_datetime(data['time'])
            mask = (data['time'] > '5/11/2022 11:20:00') & (data['time'] <= now)
            data = data.loc[mask]
            field_value = "European History"
            total_attempts = data["correct_status"].count()

            field_data = data[data['field'] == 'European History']

            correct_data = field_data[field_data['correct_status'] == 'correct']

            incorrect_data = field_data[field_data['correct_status'] == 'incorrect']

            st.write('In May 2022, GPT-3  correctly answered {} out of {} European history questions, for a {:.2f}% accuracy rate. Click [here](https://github.com/Dr-Hutchinson/gpt-3_history_benchmark_results/blob/main/european_history_benchmark_results.csv) to see the results.'.format(len(correct_data), len(field_data), len(correct_data)/len(field_data)*100))
            st.write("Below is GPT-3's accuracy on all questions to date.")
            st.bar_chart(data2['correct_status'].value_counts())

        def world_history_data():
            sh1 = gc.open('world_history_benchmark_results')
            sh2 = gc.open('benchmark_tests')
            wks1 = sh1[0]
            wks2 = sh2[0]
            now = dt.now()
            data = wks1.get_as_df(has_header=True, index_col=None)
            data2 = wks2.get_as_df(has_header=True, index_col=None)
            data['time'] = pd.to_datetime(data['time'])
            mask = (data['time'] > '5/11/2022 11:20:00') & (data['time'] <= now)
            data = data.loc[mask]
            field_value = "World History"
            total_attempts = data["correct_status"].count()

            field_data = data[data['field'] == 'World History']

            correct_data = field_data[field_data['correct_status'] == 'correct']

            incorrect_data = field_data[field_data['correct_status'] == 'incorrect']

            st.write('In May 2022, GPT-3 correctly answered {} out of {} World history questions, for a {:.2f}% accuracy rate. Click [here](https://github.com/Dr-Hutchinson/gpt-3_history_benchmark_results/blob/main/world_history_benchmark_results.csv) to see the results.'.format(len(correct_data), len(field_data), len(correct_data)/len(field_data)*100))
            st.write("Below is GPT-3's accuracy on all questions to date.")
            st.bar_chart(data2['correct_status'].value_counts())

        if st.session_state.field == "U.S. History":
            us_history_data()
        elif st.session_state.field == "European History":
            euro_history_data()
        else:
            world_history_data()
