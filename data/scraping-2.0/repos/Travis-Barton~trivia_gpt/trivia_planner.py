import streamlit as st
import pandas as pd
from utils.language_model_tools import fact_check_question, question_generator, fix_question, grade_responses, \
    aquestion_generator, async_fact_check, async_fix_question, async_fix_and_check_question
from utils.firebase_tools import get_db
import random as rd
import string
import uuid
from langchain.callbacks import StreamlitCallbackHandler
import streamlit_survey as ss
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
st.set_page_config(layout="wide",
                   page_title="AI Trivia",
                   page_icon="✍️")

db = get_db()


def check_combination_exists(combination):
    games_ref = db.collection('games')
    # Query for the specific combination
    results = games_ref.where('game_id', '==', combination).limit(1).get()

    # If any results are returned, the combination exists
    return len(results) > 0


def generate_game_id():
    combination = ''.join(rd.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    if check_combination_exists(combination):
        return generate_game_id()
    return combination


# @st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def clear_cache():
    st.session_state.data = pd.DataFrame()


# In your main coroutine
async def process_rows():
    tasks = []

    for i, row in st.session_state.data.iterrows():
        tasks.append(async_fact_check(row.question, row.answer, row.category))

    # Gather results from all tasks
    results = await asyncio.gather(*tasks)

    return results


async def process_and_update_row(i, row):
    if not row.fact_check:
        st.write(f'fixing question {i + 1}')
        result = await async_fix_and_check_question(row.question, row.answer, row.category, row.explanation,
                                                    st.session_state.data.question.tolist())
        st.session_state.data.loc[i, 'question'] = result['question']
        st.session_state.data.loc[i, 'answer'] = result['answer']
        st.session_state.data.loc[i, 'category'] = result['category']
        st.session_state.data.loc[i, 'explanation'] = result['explanation']
        st.session_state.data.loc[i, 'fact_check'] = result['fact_check']


async def process_and_update_rows():
    tasks = []

    for i, row in st.session_state.data.iterrows():
        tasks.append(process_and_update_row(i, row))

    await asyncio.gather(*tasks)


def main():
    st.markdown('''
    # Welcome to AI Trivia! 🎉
    ## A fun AI power application that generates questions and grades answers in real-time!
    ''')
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame()

    with st.sidebar:
        st.title("Download the questions template")
        df = pd.DataFrame(columns=['questions', 'answers', 'category'])
        st.download_button(
            label="Download data as CSV",
            data=convert_df(df),
            file_name='questions_template.csv',
            mime='text/csv',
            key='template'
        )
        st.write('---')
        st.title("Try a Demo Dataset")
        demo = pd.read_csv('demo.csv', index_col=0)
        st.download_button(
            label="Download data as CSV",
            data=convert_df(demo),
            file_name='questions_template.csv',
            mime='text/csv',
            key='demo'
        )
        st.write('---')
    question_tab, new_game, games_tab, how_to_use = st.tabs(["Questions", "Creating Games", "Games", "How to use"])
    with question_tab:
        st.header("Questions")
        with st.sidebar:
            st.title("Settings ⚙️")
            hide_answers = st.checkbox("hide questions", value=False)
        if st.checkbox('Use Your Own Data', on_change=clear_cache):
            file = st.file_uploader("Upload file", type=["csv"], on_change=clear_cache)
            if file is None:
                st.stop()
            if (st.session_state.data.empty):
                st.session_state.data = pd.read_csv(file, index_col=False)
        else:
            catagories = st.text_input("Enter Categories seperated by a comma", on_change=clear_cache)
            catagories = catagories.split(',')
            catagories = [cat.strip() for cat in catagories if cat.strip() != '']
            if len(catagories) > 5:
                st.warning('catagories are currently limited to 5')
                catagories = catagories[:5]
            col1, col2, col_3 = st.columns(3)
            with col1:
                question_count = st.number_input("Enter number of questions per category", value=4, max_value=10,
                                                 min_value=1, on_change=clear_cache)
            with col2:
                difficulty = st.selectbox("Select Difficulty", ['Hard', 'Medium', 'Easy'])
            with col_3:
                temperature = st.selectbox("Select a Randomness", ["Strict", "Slightly Random", "Vary Random"])
                if temperature in ['Slightly Random', 'Vary Random']:
                    st.info('Increasing the randomness will increase the change of wrong answers. Always use the AI fact check to verify these answers.')
                temperature = 0 if temperature == 'Strict' else 0.8 if temperature == 'Slightly Random' else 1.2
            if st.session_state.data.empty:
                st.session_state.data = pd.read_csv('template.csv')
            cols = st.columns(2)
            if st.button("Generate Questions"):
                with st.status("Preparing data...", expanded=True, state='running') as status:
                    st.write('Generating questions...')
                    result = asyncio.run(aquestion_generator(catagories, question_count, difficulty, temperature=temperature))
                    # result = question_generator(categories=catagories, question_count=question_count, difficulty=difficulty, st_status=status)
                    # for category, questions in result.items():
                    for _, results in result.items():
                        for category, questions in results.items():
                            _ = pd.json_normalize(questions)
                            if len(_) > question_count:
                                _ = _.sample(question_count)
                            _['category'] = category
                            st.session_state.data = pd.concat([st.session_state.data, _], axis=0)
                    st.session_state.data = st.session_state.data.dropna(how='all')
                    st.session_state.data = st.session_state.data.reset_index(drop=True)
                    st.session_state.data = st.session_state.data[['question', 'answer', 'category']]
                    st.session_state.data['fact_check'] = None
                    st.write("Fact checking questions...")
                    result = asyncio.run(process_rows())
                    for i, row in st.session_state.data.iterrows():
                        response_dict = [res for res in result if res['question'] == row.question][0]
                        status.update(label=f"Fact checking questions... {i + 1}/{len(st.session_state.data)}",
                                      state='running')
                        if response_dict['fact_check']:
                            st.session_state.data.loc[i, 'fact_check'] = response_dict['fact_check']
                            st.session_state.data.loc[i, 'explanation'] = ""
                        else:
                            st.session_state.data.loc[i, 'fact_check'] = response_dict['fact_check']
                            st.session_state.data.loc[i, 'explanation'] = response_dict['explanation']

            else:
                print(st.session_state.data)

        with st.sidebar:
            fact_check_button = st.button("Run AI Fact-Check")
        if fact_check_button:
            with st.status("Fact checking questions...", expanded=True, state='running') as status:
                result = asyncio.run(process_rows())
                for i, row in st.session_state.data.iterrows():
                    response_dict = [res for res in result if res['question'] == row.question][0]
                    status.update(label=f"Fact checking questions... {i + 1}/{len(st.session_state.data)}",
                                  state='running')
                    if response_dict['fact_check']:
                        st.session_state.data.loc[i, 'fact_check'] = response_dict['fact_check']
                        st.session_state.data.loc[i, 'explanation'] = ""
                    else:
                        st.session_state.data.loc[i, 'fact_check'] = response_dict['fact_check']
                        st.session_state.data.loc[i, 'explanation'] = response_dict['explanation']
        if ('fact_check' in st.session_state.data.columns) and any(st.session_state.data.fact_check == False):
            st.warning("Some questions failed fact checking. Please review the questions and answers.")
        if hide_answers:
            temp_data = st.session_state.data.copy()
            temp_data['answer'] = 'xxx'
            temp_data['question'] = 'xxx'
            temp_data['explanation'] = 'xxx'
            st.dataframe(temp_data, hide_index=True)
        else:
            st.session_state.data = st.data_editor(st.session_state.data, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download data as CSV",
                data=convert_df(st.session_state.data),
                file_name='questions.csv',
                mime='text/csv',
            )
        with col2:
            with st.sidebar:
                replace_button = st.button('AI Fix Answers')
            if replace_button:
                with st.status("Finding new questions...", expanded=True, state='running') as status:
                    asyncio.run(process_and_update_rows())
                    st.experimental_rerun()


    with new_game:
        with st.sidebar:
            st.title('Advanced Settings')
            st.write(
                '**Note:** These settings are for advanced users only. Please do not change these settings unless you know what you are doing.')
            st.write(
                "\n\nThese changes must be made prior to starting the game. Once the game has started, these settings cannot be changed.")
            show_rational = st.checkbox('Show Rational', value=False)
            include_answers = st.checkbox('Include Answers', value=False)

        data_upload = st.file_uploader("Upload file", type=["csv"], key='data_upload')
        if data_upload and st.button('Start Game'):
            if 'csv' in data_upload.type:
                df = pd.read_csv(data_upload)
            elif 'xlsx' in data_upload.type:
                df = pd.read_excel(data_upload)

            game_id = generate_game_id()
            questions_ref = db.collection(u'questions')
            question_ids = []
            # Iteratively add questions and keep track of their IDs
            for idx, row in df.iterrows():
                question_id = str(uuid.uuid4())
                question_ids.append(question_id)
                questions_ref.document(question_id).set({
                    'question': row['question'],
                    'answer': row['answer'],
                    'category': row['category'],
                    'game_id': game_id,
                    'question_id': question_id,
                    'revealed': False,
                    'waiting_screen': False,
                    'created_at': datetime.now(),
                    'modified_at': datetime.now(),
                    # 'correct': False,
                    'order': idx,
                })

            # Set up new game
            game_ref = db.collection(u'games').document(game_id)
            game_ref.set({
                'game_id': game_id,
                'user_ids': [],
                'question_ids': question_ids,
                'show_answers': False,
                'created_at': datetime.now(),
                'scores': {},
                'waiting_screen': False,
                'include_answers': include_answers,
                'show_rational': show_rational,
            })
            st.write(f'Game {game_id} Started')


    with games_tab:

        st.title("Lets Play Trivia!")
        st.write('---')
        if st.session_state.data.empty:
            st.warning("Please upload or generate questions first.")
        else:

            survey = ss.StreamlitSurvey()
            with st.sidebar:
                if st.button('Reset Game'):  # delete self.data_name not in st.session_state: from session state
                    st.session_state.data_name = None
                    st.session_state[survey.data_name] = {}
                    pages = survey.pages(len(st.session_state.data),
                                         on_submit=lambda: grade_responses(survey.to_json()))
            pages = survey.pages(len(st.session_state.data), on_submit=lambda: grade_responses(survey.to_json()))
            with pages:
                page_count = pages.current
                question = st.session_state.data.loc[page_count, 'question']
                answer = st.session_state.data.loc[page_count, 'answer']
                category = st.session_state.data.loc[page_count, 'category']
                st.markdown(f"Question {page_count + 1}/{len(st.session_state.data)}")
                st.write(st.session_state.data.loc[page_count, 'question'])
                st.write('---')
                survey.text_input(label=f"{question} || {answer} || {category}",
                                  label_visibility='collapsed')
                st.write('---')

    with how_to_use:
        st.title("How to use")
        st.subheader('How to generate trivia!')
        with st.expander("Click here to see how to generate trivia"):
            with open('how to add free form.mp4', 'rb') as f:
                video = f.read()
                st.video(video)

        st.subheader('How to bring your own questions!')
        with st.expander("Click here to see how to bring your own questions"):
            with open('how to add questions.mp4', 'rb') as f:
                video = f.read()
                st.video(video)

        st.subheader('How to play trivia!')
        with st.expander("Click here to see how to play trivia"):
            with open('how to play.mp4', 'rb') as f:
                video = f.read()
                st.video(video)





main()
