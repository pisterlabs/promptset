import streamlit as st
import requests
# import pymongo
import random
# import openai

# API URLs
QUESTION_API_URL = "https://axisapi.onrender.com/Question"
ASSESS_API_URL = "https://axisapi.onrender.com/Assess"
SUBMIT_API_URL= "https://axisapi.onrender.com/submit"

# Fixed job description
FIXED_JOB_DESCRIPTION = "AI Research Scientist"

# Function to get interview questions
def get_interview_questions():
    params = {
        'description': FIXED_JOB_DESCRIPTION
    }

    response = requests.get(QUESTION_API_URL, params=params)
    data = response.json()
    return data['questions']

# Function to calculate and display the score
def calculate_and_display_score(questions, answers, email):
    params = {
        'questions': questions,
        'answers': answers,
        'email': email
    }

    response = requests.get(ASSESS_API_URL, params=params)
    if response.status_code == 200:
        score_data = response.json()
        return score_data['score']
    else:
        return None
    
def final_submission(email, score, job_id):
    params = {
        'email': email,
        'score': score,
        'job_id': job_id
    }

    response= requests.post(SUBMIT_API_URL, params=params)
    if response.status_code == 200:
        print('POST request successful')
        print('Response:', response.text)
    else:
        print('POST request failed')
        print('Status code:', response.status_code)
        print('Response:', response.text)


def initialize_session_state():
    if "questions" not in st.session_state:
        st.session_state.questions = get_interview_questions()

    if "question_index" not in st.session_state:
        st.session_state.question_index = 0

    if "answer1" not in st.session_state:
        st.session_state.answer1 = ""

    if "answer2" not in st.session_state:
        st.session_state.answer2 = ""

    if "answer3" not in st.session_state:
        st.session_state.answer3 = ""

    if "answer4" not in st.session_state:
        st.session_state.answer4 = ""

    if "answer5" not in st.session_state:
        st.session_state.answer5 = ""

    if "score1" not in st.session_state:
        st.session_state.score1 = 0

    if "score2" not in st.session_state:
        st.session_state.score2 = 0

    if "score3" not in st.session_state:
        st.session_state.score3 = 0

    if "score4" not in st.session_state:
        st.session_state.score4 = 0

    if "score5" not in st.session_state:
        st.session_state.score5 = 0

def main():
    st.title("xsBot.ai")
    
    st.write(f"Role: {FIXED_JOB_DESCRIPTION}")
    
    initialize_session_state()

    email = st.text_input("Enter your email:")

    if st.button("Submit Email"):
        st.session_state.email = email

    questions = st.session_state.questions
    
    answer1 = st.text_input(f"Q1: {questions[0]}", key="answer_0")
    
    if st.button("Submit Answer 1"):
        st.session_state.answer1=answer1
        st.session_state.score1= calculate_and_display_score([questions[0]], [st.session_state.answer1], st.session_state.email)
        st.write(f"Score for Q1: {st.session_state.score1}")


    answer2 = st.text_input(f"Q2: {questions[1]}", key="answer_1")
    
    if st.button("Submit Answer 2"):
        st.session_state.answer2=answer2
        st.session_state.score2= calculate_and_display_score([questions[1]], [st.session_state.answer2], st.session_state.email)
        st.write(f"Score for Q2: {st.session_state.score2}")


    answer3 = st.text_input(f"Q3: {questions[2]}", key="answer_2")
    
    if st.button("Submit Answer 3"):
        st.session_state.answer3=answer3
        st.session_state.score3= calculate_and_display_score([questions[2]], [st.session_state.answer3], st.session_state.email)
        st.write(f"Score for Q3: {st.session_state.score3}")

    answer4 = st.text_input(f"Q4: {questions[3]}", key="answer_3")

    if st.button("Submit Answer 4"):
        st.session_state.answer4=answer4
        st.session_state.score4= calculate_and_display_score([questions[3]], [st.session_state.answer4], st.session_state.email)
        # st.write(f"Score for Q4: {st.session_state.score4}")   

    answer5 = st.text_input(f"Q5: {questions[4]}", key="answer_4")

    if st.button("Submit Answer 5"):
        st.session_state.answer5=answer5
        st.session_state.score5= calculate_and_display_score([questions[4]], [st.session_state.answer5], st.session_state.email)
        # st.write(f"Score for Q5: {st.session_state.score5}") 

    if st.button("Generate Score"):
            final_score= st.session_state.score1 + st.session_state.score2 + st.session_state.score3 + st.session_state.score4 + st.session_state.score5
            # final_score= st.session_state.answer1 + st.session_state.answer2 + st.session_state.answer3 + st.session_state.answer4 + st.session_state.answer5
            st.success(f"Your Score: {final_score}/50")
            st.write("You can now close the tab")
            final_submission(st.session_state.email, final_score, "1467")

    
    # if "email" in st.session_state:
    #     question_index = st.session_state.question_index
    #     questions = st.session_state.questions

    #     final_score = 0

    #     if question_index < len(questions):
    #         st.write(f"Current index: {question_index}")
    #         st.session_state.question_index += 1  # Increment question index
            
    #         current_question = questions[question_index]
            
    #         # Use a unique key for each text_area to avoid rendering issues
    #         answer = st.text_input(f"Q{question_index+1}: {current_question}", key=f"answer_{question_index}")
            
    #         if st.button("Next"):
    #             st.session_state.answers.append(answer)  # Append answer to the list
        
    #             # Calculate the score for the current question
    #             score = calculate_and_display_score([current_question], [answer], st.session_state.email)
        
    #             # Display the calculated score for the current question
    #             if score is not None:
    #                 st.write(f"Score for Q{question_index+1}: {score}")
    #                 final_score = final_score + score
        
    #             st.experimental_rerun()  # Force UI update
    #     else:
    #         st.write("All questions answered. Click 'Generate Score' to see your score.")

    #     if st.button("Generate Score"):
    #         answers = st.session_state.answers
    #         st.success(f"Your Score: {final_score}")
    #         st.write("You can now close the tab")

if __name__ == "__main__":
    main()