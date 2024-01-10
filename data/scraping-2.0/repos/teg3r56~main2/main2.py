import streamlit as st
import streamlit.components.v1 as components
import openai
import ast
import random
import time
import math
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

def calculate_delay(percent_complete, number_of_items):
    
    base_time = 0.02  
    incremental_time = 0.025  

    time_delay = base_time + ((number_of_items - 1) * incremental_time)

    if percent_complete > 50:
        time_delay *= (1 + (percent_complete - 50) / 50)
    if percent_complete > 85:
        time_delay *= (1 + (percent_complete - 85) / 15)
    if percent_complete > 95:
        time_delay *= (1 + (percent_complete - 95) / 5)
    if percent_complete > 99:
        time_delay *= (1 + (percent_complete - 99))

    return time_delay

def parse_questions(content):
    try:
        valid_questions = ast.literal_eval(content)
        if isinstance(valid_questions, list) and all(isinstance(question, tuple) and len(question) == 4 for question in valid_questions):
            return valid_questions
        else:
            st.error("The API response is not in the expected format of a list of tuples.")
            return None
    except SyntaxError as e:
        st.error(f"Syntax error while parsing content: {e}")
        return None
    except Exception as e:
        st.error(f"Error while parsing content: {e}")
        return None
        
def generate_flashcards_from_topic(topic, number_of_items):

    if 'progress_bar' not in st.session_state:
            st.session_state.progress_bar = st.progress(0)

    update_progress_bar()

    try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": f"Generate {number_of_items} flashcards on the topic of {topic}. Each flashcard should be formatted as a tuple containing a unique number, the concept, and its definition."
                    },
                    {
                        "role": "user", 
                        "content": f"Generate exactly {number_of_items} flashcards about {topic}, formatted as: [('1', 'Concept 1', 'Definition 1'), ('2', 'Concept 2', 'Definition 2')], where each tuple contains a unique number, the concept, and its definition."
                    }
                ]
            )

            content = response.choices[0].message.content.strip()
            flashcards = parse_flashcards(content)

            if flashcards:
                st.session_state.flashcards = flashcards
                st.session_state.current_flashcard_index = 0
                st.session_state.display_flashcards = True
                st.session_state.show_definition = [False] * len(flashcards)  
                return True
            else:
                st.error("Could not parse the API response into flashcards.")
                return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False
    finally:
        st.session_state.progress_bar.empty()

def parse_flashcards(content):
    try:
        valid_flashcards = ast.literal_eval(content)
        if isinstance(valid_flashcards, list) and all(isinstance(flashcard, tuple) and len(flashcard) == 3 for flashcard in valid_flashcards):
            return valid_flashcards
        else:
            st.error("The API response is not in the expected format of a list of tuples with three elements each.")
            return None
    except SyntaxError as e:
        st.error(f"Syntax error while parsing content: {e}")
        return None
    except Exception as e:
        st.error(f"Error while parsing content: {e}")
        return None

def update_progress_bar():
    progress = 0
    max_delay = 0.25  # maximum delay at 90% progress
    while progress < 90:
        delay_factor = (progress / 90) ** 2  # exponential growth factor
        sleep_time = max_delay * delay_factor
        time.sleep(sleep_time)
        progress += 1
        st.session_state.progress_bar.progress(progress)

def generate_questions_from_topic(topic, number_of_items):

    if 'progress_bar' not in st.session_state:
            st.session_state.progress_bar = st.progress(0)

    update_progress_bar()

    try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": f"Generate a list of multiple-choice questions with answers and explanations on the topic of {topic}. Please provide exactly {number_of_items} questions. Format the output as a Python list, with each question as a tuple containing the question text, a list of options, the index of the correct option, and an explanation. Surround the entire list with one pair of brackets, without extra brackets around individual tuples."
                    },
                    {
                        "role": "user", 
                        "content": f"Create multiple-choice questions about {topic} with exactly {number_of_items} questions. The output should be formatted as:"
                                                        "[('question', ['options', 'options', 'options'], correct_option_index, 'explanation')] "
                                                        "Example: ["
                                                        "('How many valence electrons do elements in the Alkali metal family have?', "
                                                        "['1', '2', '3'], 0, 'Alkali metals belong to Group 1A and have 1 valence electron.'),"
                                                        "('What is the common oxidation state of Alkali metals?', ['+1', '+2', '0'], 0, "
                                                        "'Alkali metals have an oxidation state of +1 as they tend to lose one electron.')]"
                    }
                ]
            )

            content = response.choices[0].message.content.strip()

            if not content.startswith("[") or not content.endswith("]"):
                content = "[" + content.replace("]\n\n[", ", ") + "]"

            questions = parse_questions(content)

            with st.spinner('Formatting your quiz...'):
                if questions:
                    random.shuffle(questions)
                    st.session_state.questions = questions
                    st.session_state.current_question_index = 0
                    st.session_state.correct_answers = 0
                    st.session_state.display_quiz = True
                    quiz_ready = True 
                    if quiz_ready:
                        st.session_state.progress_bar.progress(100)
                    return True
                else:
                    st.error("Could not parse the API response into quiz questions.")
                    return False
    except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return False
    finally:
        st.session_state.progress_bar.empty()

if 'questions' not in st.session_state:
    st.session_state.questions = []
    st.session_state.correct_answers = 0
    st.session_state.current_question_index = 0
    st.session_state.show_next = False

if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []

def reset_display_states():
    """Reset states related to the display of quizzes and flashcards."""
    st.session_state.display_flashcards = False
    st.session_state.display_quiz = False
    st.session_state.show_results = False
    st.session_state.quiz_started = False
    st.session_state.review_ready = False
    st.session_state.show_next = False
    st.session_state.answer_submitted = False

def main_screen():
    if 'Questions_or_Flashcards' not in st.session_state:
        st.session_state.Questions_or_Flashcards = "Number of Questions"
    if 'load_next_question' not in st.session_state:
        st.session_state.load_next_question = False
    if 'display_flashcards' not in st.session_state:
        st.session_state.display_flashcards = False
    if 'choice' not in st.session_state:
        st.session_state.choice = None
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'review_ready' not in st.session_state:
        st.session_state.review_ready = False
    if 'generate_pressed' not in st.session_state:
        st.session_state.generate_pressed = False
    if 'last_answer_was_correct' not in st.session_state:
        st.session_state.last_answer_was_correct = None
    if 'last_explanation' not in st.session_state:
        st.session_state.last_explanation = ''
    if 'progress_bar_placeholder' not in st.session_state:
        st.session_state.progress_bar_placeholder = st.empty()
    if 'show_next' not in st.session_state:
        st.session_state.show_next = False
    if 'answer_submitted' not in st.session_state:
        st.session_state.answer_submitted = False
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'quiz_or_flashcard' not in st.session_state:
        st.session_state.quiz_or_flashcard = None
    
    hue_shift_square()

    apply_css_styles()

    st.title("Quiz Generator")

    #st.markdown('<p class="gpt-font">Created By Teague Coughlin</p>', unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .st-co.st-cn.st-cm.st-cl {
            border-top-color: rgb(72 255 202 / 60%);
            border-bottom-color: rgb(60 197 157 / 66%);
            border-left-color: rgb(72 255 202 / 50%);
            border-right-color: rgb(60 197 157 / 66%);
        }
        </style>""", unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .st-bt.st-bs.st-br.st-bq {
            border-top-color: rgb(72 255 202 / 30%);
            border-bottom-color: rgb(60 197 157 / 46%);
            border-left-color: rgb(72 255 202 / 30%);
            border-right-color: rgb(60 197 157 / 46%);
        }
        </style>""", unsafe_allow_html=True)
    
    topic = st.text_input("Enter the topic or notes you want to study:")
    
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False

    if 'review_ready' not in st.session_state:
        st.session_state.review_ready = False

    col1, col2 = st.columns(2)

    selected_option = st.session_state.get('selected_option', '')    

    with col1:
        st.markdown("<div class='flex-container'>", unsafe_allow_html=True)
        if st.button("Generate a Quiz", key="generate_quiz_button", 
                     on_click=lambda: st.session_state.update({'selected_option': 'quiz'})):
            st.session_state.choice = "quiz"
            st.session_state.Questions_or_Flashcards = "Number of Questions"
            st.session_state.generate_pressed = False

    with col2:
        st.markdown("<div class='flex-container'>", unsafe_allow_html=True)
        if st.button("Generate Flashcards", key="generate_flashcards_button", 
                     on_click=lambda: st.session_state.update({'selected_option': 'flashcard'})):
            st.session_state.choice = "flashcard"
            st.session_state.Questions_or_Flashcards = "Number of Flashcards"
            st.session_state.generate_pressed = False

    number_input_placeholder = st.empty()
    generate_button_placeholder = st.empty()

    if 'selected_option' in st.session_state:
        if st.session_state.selected_option == 'quiz':
            st.markdown(
                "<style>#generate_quiz_button { background-color: white !important; color: black !important; }</style>", 
                unsafe_allow_html=True)
        elif st.session_state.selected_option == 'flashcard':
            st.markdown(
                "<style>#generate_flashcards_button { background-color: white !important; color: black !important; }</style>", 
                unsafe_allow_html=True)

    if st.session_state.choice and not st.session_state.generate_pressed:
        number_of_questions = number_input_placeholder.number_input(f"{st.session_state.Questions_or_Flashcards}", min_value=1, max_value=40, value=5, key='number_of_questions')
        if generate_button_placeholder.button("Generate"):
            # reset display states before generating new content
            reset_display_states()
            st.session_state.generate_pressed = True
            if st.session_state.choice == "quiz":
                generate_questions_from_topic(topic, number_of_questions)
            elif st.session_state.choice == "flashcard":
                generate_flashcards_from_topic(topic, number_of_questions)
            number_input_placeholder.empty()
            generate_button_placeholder.empty()
            st.experimental_rerun()

    if st.session_state.get('display_quiz', False):
        display_current_question()

    if st.session_state.get('display_flashcards', False):
        display_flashcards()

    if st.session_state.get('show_results', False):
        display_results()
    
    if 'restart_quiz' in st.session_state and st.session_state.restart_quiz:
        st.session_state.quiz_started = False
        st.session_state.choice = None

def generate_quiz_or_flashcards(topic, number_of_items):
    if st.session_state.choice == "quiz":
        quiz_generated = generate_questions_from_topic(topic, number_of_items)
    elif st.session_state.choice == "flashcard":
        flashcards_generated = generate_flashcards_from_topic(topic, number_of_items)

    st.experimental_rerun()

def display_flashcards():
    total_flashcards = len(st.session_state.flashcards)
    current_index = st.session_state.current_flashcard_index
    flashcard = st.session_state.flashcards[current_index]
    concept, definition = flashcard[1], flashcard[2]
    showing_definition = st.session_state.show_definition[current_index]

    # button to "flip" the card
    st.button('Show Definition' if not showing_definition else 'Show Concept',
              key=f'flip_{current_index}',
              on_click=toggle_definition,
              args=(current_index,))

    concept_color = "#dbdcdd"  # concept text color
    html_content = f"""
    <div style="
        border: 2px solid #2a2e36;
        border-radius: 7px;
        padding: 20px;
        font-size: 20px;
        font-color: #f6f6f7;
        background-color: #262730;
        text-align: center;
        box-shadow: 0 2px 4px 0 rgba(155,155,255,0.2);
    ">
        <h2 style="color: {concept_color if not showing_definition else '#dbdcdd'};">{'Concept' if not showing_definition else 'Definition'}</h2>
        <p style="
            font-size: 18px;
            color: #b1b3b5;
        ">{concept if not showing_definition else definition}</p>
    </div>
    """
    components.html(html_content, height=200)

    # pages pagination
    if total_flashcards > 1:  
        pagination_cols = [1] + [1] * total_flashcards + [1]
        cols = st.columns(pagination_cols)

        if cols[0].button("Previous"):
            change_flashcard(current_index - 1)

        for i in range(total_flashcards):
            index_button_key = f"page_{i}"
            # add the current page number to the key to make each key unique for the index
            if cols[i + 1].button(f"{i + 1}", key=index_button_key):
                change_flashcard(i)

        if cols[-1].button("Next"):
            change_flashcard(current_index + 1)

def toggle_definition(index):
    st.session_state.show_definition[index] = not st.session_state.show_definition[index]

def change_flashcard(new_index):
    new_index = max(0, min(new_index, len(st.session_state.flashcards) - 1))
    if new_index != st.session_state.current_flashcard_index:
        st.session_state.current_flashcard_index = new_index
        st.session_state.show_definition = [False] * len(st.session_state.flashcards)
        # force rerun to immediately reflect the change
        st.experimental_rerun()

def check_answer(option, options, correct_answer_index, explanation):
    if options.index(option) == correct_answer_index:
        st.session_state.correct_answers += 1
        st.session_state.last_answer_was_correct = True
    else:
        st.session_state.last_answer_was_correct = False
    
    st.session_state.last_explanation = explanation
    st.session_state.answer_submitted = True

if 'submit_button_placeholder' not in st.session_state:
    st.session_state.submit_button_placeholder = st.empty()
if 'next_button_placeholder' not in st.session_state:
    st.session_state.next_button_placeholder = st.empty()
if 'message_placeholder' not in st.session_state:
    st.session_state.message_placeholder = st.empty()
if 'explanation_placeholder' not in st.session_state:
    st.session_state.explanation_placeholder = st.empty()

def display_current_question():
    if st.session_state.get('display_quiz', False) and st.session_state.questions:
        question_tuple = st.session_state.questions[st.session_state.current_question_index]
        question, options, correct_answer_index, explanation = question_tuple
        
        st.write(question)
        
        selected_option = st.radio("Choose the correct answer:", options, key=f"option{st.session_state.current_question_index}")

        # button/explanation placeholders
        submit_button_placeholder = st.empty()
        next_button_placeholder = st.empty()
        message_placeholder = st.empty()
        explanation_placeholder = st.empty()

        # answer has not been submitted, show the button
        if not st.session_state.get('answer_submitted', False):
            if submit_button_placeholder.button("Submit Answer"):
                # answer has been submitted
                st.session_state.answer_submitted = True
                check_answer(selected_option, options, correct_answer_index, explanation)
                if st.session_state.last_answer_was_correct:
                    message_placeholder.success("Correct!")
                else:
                    message_placeholder.error("Incorrect!")
                explanation_placeholder.info(explanation)
                submit_button_placeholder.empty()
                if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                    next_button_placeholder.button("Next Question", on_click=next_question)
                else:
                    next_button_placeholder.button("Review", on_click=handle_quiz_end)
        else:
            if st.session_state.last_answer_was_correct:
                message_placeholder.success("Correct!")
            else:
                message_placeholder.error("Incorrect!")
            explanation_placeholder.info(explanation)
            if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                next_button_placeholder.button("Next Question", on_click=next_question)
            elif st.session_state.current_question_index == len(st.session_state.questions) - 1:
                next_button_placeholder.button("Review", on_click=handle_quiz_end)

def display_results():
    correct_answers = st.session_state.correct_answers
    total_questions = len(st.session_state.questions)
    letter_grade = get_letter_grade(correct_answers, total_questions)
    grade_color_style = f"color: {grade_color[letter_grade]};"
    score = f"{correct_answers} out of {total_questions}"
    st.markdown(f"Quiz Finished! You got {score} correct. Your grade: <span style='{grade_color_style}'>{letter_grade}</span>", unsafe_allow_html=True)
    if st.button("Restart Quiz"):
        restart_quiz()  

def restart_quiz():
    st.session_state.show_results = False
    st.session_state.quiz_started = False
    st.session_state.generate_pressed = False  
    st.session_state.choice = None  

    # reset the quiz state
    st.session_state.questions = random.sample(st.session_state.questions, len(st.session_state.questions))
    st.session_state.current_question_index = 0
    st.session_state.correct_answers = 0
    st.session_state.show_next = False
    st.session_state.answer_submitted = False
    st.session_state.display_quiz = True

    st.session_state.last_answer_was_correct = None
    st.session_state.last_explanation = ''
    
    st.experimental_rerun()

def next_question():
    if st.session_state.current_question_index < len(st.session_state.questions) - 1:
        st.session_state.current_question_index += 1
        st.session_state.answer_submitted = False
    else:
        handle_quiz_end()

    st.session_state.load_next_question = True

def check_answer(selected_option, options, correct_answer_index, explanation):
    if options.index(selected_option) == correct_answer_index:
        st.session_state.correct_answers += 1
        st.session_state.last_answer_was_correct = True
    else:
        st.session_state.last_answer_was_correct = False
    st.session_state.last_explanation = explanation
    
    st.session_state.answer_submitted = True
    
def get_letter_grade(correct, total):
    if total == 0: return 'N/A'  # division by zero 
    percentage = (correct / total) * 100
    if percentage >= 90: return 'A'
    elif percentage >= 80: return 'B'
    elif percentage >= 70: return 'C'
    elif percentage >= 60: return 'D'
    else: return 'F'
        
# color dictionary
grade_color = {
    'A': '#4CAF50',  # Green
    'B': '#90EE90',  # Light Green
    'C': '#FFC107',  # Amber
    'D': '#FF9800',  # Orange
    'F': '#F44336',  # Red
}

def handle_quiz_end():
    st.session_state.quiz_started = False
    st.session_state.display_quiz = False
    st.session_state.show_results = True

def hue_shift_square():
    css_string = """
            .hue-block {{
                position: absolute;
                height: auto;
                width: 100%;
                padding: 1rem;
                border-radius: 15px;
                background: black;
                background-size: 1000% 1000%;
                opacity: 0.7;
                animation: gradient-shift 30s linear infinite;
                background-image: linear-gradient(
                    120deg,
                    #0000FF,
                    #0000E0,
                    #4682B4,
                    #0000A0,
                    #000080,
                    #000060,
                    #000040,
                    #000020,
                    #000010,
                    #0000FF
                );
                z-index: 1;
            }}

            @keyframes gradient-shift {{
                0%, 100% {{ background-position: 100% 50%; }}
                50% {{ background-position: 0% 50%; }}
            }}
    """
    # Use the custom component to apply the CSS
def apply_css_styles():
    styles = """
    <style>
    .st-by .st-ci {
        caret-color: rgb(255, 255, 255);
        color: rgb(255, 255, 255);
    }
    h1 {
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 700;
        color: rgb(232 232 232);
        padding: 1.25rem 0px 1rem;
        margin: 0px;
        line-height: 1.2;
    }
    .st-bw {
        color: rgb(249, 250, 250);
    }
    .st-emotion-cache-ue6h4q {
        color: rgb(250, 250, 250);
    }
    .st-bv {
        background-color: rgb(91 192 255 / 57%);
    }
    .st-fa {
        background-color: #8a9cb800;
    }
    .st-emotion-cache-bdfrcy {
        width: 95%;
        padding-bottom: 1px;
        line-height: normal;
        color: rgb(253 242 242);
    }
    .st-emotion-cache-1kt38h1 {
        transition: background-color 0.9s ease; 
        display: inline-flex;
        -webkit-box-align: center;
        align-items: center;
        -webkit-box-pack: center;
        justify-content: center;
        font-weight: 100;
        text-size-adjust: 100%;
        padding: 0.25rem 0.75rem;
        border-radius: 0.7rem;
        min-height: 38.4px;
        margin: 0px;
        line-height: 1.6;
        color: #fafafa;
        width: auto;
        user-select: none;
        background-color: rgb(14 50 239 / 40%);
        border: 1px solid rgb(255, 255, 255 0.4);
    }
    .st-emotion-cache-19j208r {
        position: fixed;
        top: 0px;
        left: 0px;
        right: 0px;
        height: 2.875rem;
        background: #161b22;
        outline: none;
        z-index: 999990;
        display: block;
    }
    .st-emotion-cache-1pfffnf {
        position: absolute;
        opacity: 0.9;
        background: linear-gradient(120deg, #006994, darkblue);
        color: rgb(250, 250, 250);
        inset: 0px;
        overflow: hidden;
    }
    .st-emotion-cache-1kt38h1:hover {
        border-color: rgb(17 22 24);
        color: rgb(255, 255, 255);
        box-shadow: 0px 0px 2px 1px rgba(0,0,224,0.00);
    }
    .st-emotion-cache-1kt38h1:focus:not(:active) {
        background-color: rgb(212, 228, 250, 0.45);
        border-color: rgb(17 22 24);
        color: rgb(0 0 0);
        box-shadow: 0px 0px 2px 1px rgba(255,255,224,0.00);
    }
    .st-emotion-cache-1kt38h1:active {
        opacity: 0.9;
        backdrop-filter: blur(10px);
        background-color: rgb(212, 228, 250, 0.45);
        border-color: rgb(17 22 24);
        color: rgb(0 0 0);
    }
    
    .st-emotion-cache-1aof7ch.focused {
        border-top-color: rgb(72 255 202 / 30%);
        border-bottom-color: rgb(60 197 157 / 46%);
        border-left-color: rgb(72 255 202 / 30%);
        border-right-color: rgb(60 197 157 / 46%);
    }

    .st-emotion-cache-1aof7ch {
        border-top-color: rgb(72 255 202 / 30%);
        border-bottom-color: rgb(60 197 157 / 46%);
        border-left-color: rgb(72 255 202 / 30%);
        border-right-color: rgb(60 197 157 / 46%);
    }
    .st-emotion-cache-bdfrcy {
        width: 100%;
        padding-bottom: 1px;
        line-height: normal;
        color: rgb(0 0 0);
    }
    .st-emotion-cache-1hgxyac {
        margin: 0px;
        border: none;
        height: 100%;
        display: flex;
        -webkit-box-align: center;
        align-items: center;
        width: 32px;
        -webkit-box-pack: center;
        justify-content: center;
    }
    .gpt-font { font sans-serif; font-size:1rem; margin-left: 3px; opacity: 1; margin-top: -10px; }
    div.stButton > button:first-child { width: 100%; }
    </style>
    """
    st.markdown(styles, unsafe_allow_html=True)

if __name__ == "__main__":
    main_screen()
