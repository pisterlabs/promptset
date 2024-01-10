import streamlit as st
import random
import json
from langchain.chat_models import ChatOpenAI
from gpt_functions import (mc_questions_json, 
                           fitb_generate, 
                           find_word_positions,
                           same_meaning)

if 'sample_text' not in st.session_state:
    print("---> Resetting sample_text")
    st.session_state['sample_text'] = "None"
    
if 'count' not in st.session_state:
	st.session_state.count = 0

def increment_counter():
	st.session_state.count += 1
    
st.title("Reading Comprehension")
st.subheader("Pass your English test with the power of AI")

test_choice = st.radio("What are you studying for?",
                       options=["IELTS", "CELPIP"])

st.text(f"We will help you study for {test_choice}")

# @st.cache_data()
def reading_task(sample):
    task_container.info("Read the article below to start your assessment")
    # ar_container.subheader(f"Academic Reading test 1 - section 1 practice test")
    ar_container.caption(
        "This is the first section of your IELTS Reading test. \
            You should spend about 20 minutes on Questions 1–13, \
                which are based on Reading Passage 1 below.")
    ar_container.write(sample.read())
    return sample.read()

if st.button("Get reading task", on_click=increment_counter):
    if st.session_state.count > 5:
        st.session_state.count = 1
        
    n = st.session_state.count
    st.text(f"Reading Task #: {n}")
    sample = open(f'tasks/{test_choice}_reading_tasks/sample_{n}.txt', 'r',
              encoding="utf8")
    task_container = st.container()
    ar_container = st.container()
    reading_task(sample)
    

    st.session_state.sample_text = open(f'tasks_dataset/{test_choice}_reading_tasks/sample_{n}.txt', 'r',
            encoding="utf8").read()
    with st.expander("show sample_text"):
        st.write(st.session_state.sample_text)
    
else:
    st.markdown("Push button")
    with st.expander("Show last reading task"):
        # st.write(st.session_state.sample_text)
        st.write(st.session_state.sample_text)

#####################################
## MULTIPLE CHOICE QUESTIONS ##
##################################

n_input = st.number_input("Number of questions to generate.", 
                              min_value=1,
                              max_value=10, 
                              value=2)
n_int = int(n_input)

@st.cache_data(show_spinner = f"Generating {n_int} questions from summary...")
def generate_questions(text, n_int):
    return mc_questions_json(text, n=n_int)


# maps integers to letters for keeping track of answers
int2letter = {0:"A", 1:"B", 2:"C", 3:"D"}
letter2int = {"A":0, "B":1, "C":2, "D":3}
if "current_question" not in st.session_state:
    st.session_state.questions_json = None
    st.session_state.answers = {} 
    st.session_state.current_question = 1 # keeps track of current question number
    st.session_state.questions = [] 
    st.session_state.right_answers = 0 # count of right answers
    st.session_state.wrong_answers = 0 # count of wrong answers

def reset_answers():
    st.session_state.answers = {} 
    st.session_state.right_answers = 0 # count of right answers
    st.session_state.wrong_answers = 0 # count of wrong answers
    st.session_state.current_question = 1 # keeps track of current question number

if st.button("Generate Multiple Choice Questions"):
    questions_json = generate_questions(st.session_state.sample_text, n_int)
    st.session_state.questions_json = questions_json
    
def display_question():
    # Handle first case
    questions_json = st.session_state.questions_json
    if len(st.session_state.questions) == 0:
        try:
            first_question = questions_json['questions'][0]
        except Exception as e:
            st.error(e)
            return
        st.session_state.questions.append(first_question)

    # Disable the submit button if the user has already answered this question
    submit_button_disabled = st.session_state.current_question in st.session_state.answers

    # Get the current question from the questions list
    question = st.session_state.questions[st.session_state.current_question-1]

    # Display the question prompt
    st.subheader(f"{st.session_state.current_question}. {question['question']}")

    # Use an empty placeholder to display the radio button question_container
    question_container = st.empty()

    # Display the radio button question_container and wait for the user to select an answer
    user_answer = question_container.radio("Please select an answer:", 
                                            question["options"],
                                            key=st.session_state.current_question)

    # Display the submit button and disable it if necessary
    submit_button = st.button("Submit", disabled=submit_button_disabled)

    # If the user has already answered this question, display their previous answer
    if st.session_state.current_question in st.session_state.answers:
        user_choice = st.session_state.answers[st.session_state.current_question]
        question_container.radio(
            "Your answer:",
            question["options"],
            key=float(st.session_state.current_question),
            index=letter2int[user_choice],
        )

    answer_index = question["options"].index(user_answer) # answer_index is an int (1-4)
    def count_answer():
        if user_answer == question["correct_answer"]:  # match the whole answer string
            st.session_state.right_answers += 1
        else:
            st.session_state.wrong_answers += 1

    def show_answer():
        # Check if the user's answer is correct and update the score
        if user_answer == question["correct_answer"]:
            st.write("Correct!")
        else:
            st.write(f"Sorry, the correct answer was {question['correct_answer']}.")

        # Show an expander with the explanation of the correct answer
        with st.expander("Explanation"):
            st.write(question["explanation"])

    # If the user clicks the submit button, check their answer and show the explanation
    if submit_button:
        # Record the user's answer in the session state
        st.session_state.answers[st.session_state.current_question] = int2letter[answer_index]
        st.caption(f"You submitted choice {int2letter[answer_index]}")
        count_answer()
        show_answer()

    elif submit_button_disabled:
        show_answer()

def next_question():
    questions_json = st.session_state.questions_json
    # Move to the next question in the questions list
    if st.session_state.current_question == n_int:
        st.caption("No more questions")
        return
    
    st.session_state.current_question += 1

    # If we've reached the end of the questions list, get a new question
    if st.session_state.current_question > len(st.session_state.questions) - 1:
        try:
            next_question = questions_json['questions'][st.session_state.current_question-1]
        except Exception as e:
            st.error(e)
            st.session_state.current_question -= 1
            return
        st.session_state.questions.append(next_question)
        # st.experimental_rerun()
        
# Define a function to go to the previous question
def prev_question():
    # Move to the previous question in the questions list
    if st.session_state.current_question > 1:
        st.session_state.current_question -= 1

# Create a 3-column layout for the Prev/Next buttons and the question display
col1, col2, col3 = st.columns([1, 4, 1])

# Add a Prev button to the left column that goes to the previous question
with col1:
    if col1.button("⬅️ Prev"):
        prev_question()

# Add a Next button to the right column that goes to the next questionG
with col3:
    if col3.button("Next ➡️"):
        next_question()

# Display the actual quiz question
with col2:
    display_question()
    
with st.sidebar: # update sidebar with newly submitted answers
    # with st.expander("Questions JSON"):
    #     st.json(questions_json)
    # with st.expander(f"Submitted answers"):
    #     st.write(st.session_state.answers)
    st.subheader("Multiple choice score:")
    # display counter for right and wrong answers
    st.caption(f"Right answers: {st.session_state.right_answers}")
    st.caption(f"Wrong answers: {st.session_state.wrong_answers}")

if st.sidebar.button("🔄 Reset quiz"):
    reset_answers()
    st.experimental_rerun()

####################################
####  END_OF_QUIZ
##################################


###########################################
#### FILL IN THE BLANK (FITB) EXERICES ####
###########################################
if "fitb" not in st.session_state:
    print("---> resetting FITB")
    st.session_state["fitb"] = None


n_fitb = st.number_input("How many FITB questions to generate.", 
                              min_value=1,
                              max_value=5, 
                              value=3)
n_fitb = int(n_fitb)

# st.subheader("Score Form")
if st.button("Generate Fill in the Blank Exercises"):
    st.session_state.fitb = fitb_generate(st.session_state.sample_text, 
                                          n= n_fitb,
                                          model='gpt-3.5-turbo')
    
with st.container():
    synonyms_allowed = st.checkbox("Synonyms Allowed")
    
    fitb_json = st.session_state.fitb
    answer_list = []
    print(type(fitb_json))
    for i, exercise in enumerate(fitb_json):
        st.markdown(f"**Exercise #{i+1}**")
        st.write(exercise["incomplete_sentence"])
        answer = st.text_input(f"Input correct word #{i+1}")
        
        if answer == "":
            pass
        elif answer == exercise["missing_word"]:
            st.success("✅ Correct")
        else:
            if synonyms_allowed and same_meaning(answer, exercise['missing_word']):
                st.info("✅ Good Enough")
            else:
                st.warning("⭕ Try again")

    with st.expander("Cheat sheet 🤫"):
        st.write(fitb_json)
        
###########################################
#### END OF FITB EXERCISES ####
###########################################
        