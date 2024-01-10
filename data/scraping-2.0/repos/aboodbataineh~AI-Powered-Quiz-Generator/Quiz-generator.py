import streamlit as st
import openai

# Initialize your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'


# Function to generate quiz questions and answers
def generate_quiz(topic, num_questions):
    questions = []
    correct_answers = []
    for _ in range(num_questions):
        prompt = f"Generate a single, detailed multiple-choice question on a {topic}, complete with four answer choices. The question should be precise, accurate, and thoroughly relevant to the topic selected. Provide the four potential answers labeled as A, B, C, and D. After the question and answers, immediately indicate which option is correct, the quiz should not include true or false questions,  and include a brief explanation to validate the answer. Ensure there is no duplication of questions and each query is unique to prevent any overlap in the quiz content."
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=150,
            n=1,

        )
        full_text = response.choices[0].text.strip()
        # Split the text into lines
        lines = full_text.split('\n')
        # Assuming the last line indicates the correct answer
        correct_answer = lines[-1]
        # Include all lines except the last one (which indicates the correct answer) in the question
        question = '\n'.join(lines[:-1])
        questions.append(question)
        correct_answers.append(correct_answer[0])

    return questions, correct_answers


# Initialize session state variables if they don't exist
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []
if 'correct_answers' not in st.session_state:
    st.session_state.correct_answers = []
if 'submit_disabled' not in st.session_state:
    st.session_state.submit_disabled = []






# Streamlit app interface
st.title("AI-Powered Quiz Generator")
topic = st.text_input("Enter your quiz topic:")
num_questions = st.number_input("Number of questions:", min_value=1, max_value=10, value=5)
st.session_state.submit_disabled = True



# Generate quiz button
if st.button("Generate Quiz"):
    st.session_state.quiz_questions, st.session_state.correct_answers = generate_quiz(topic, num_questions)
    st.session_state.user_answers = [None] * num_questions
    st.session_state.submit_disabled = False

# Display quiz questions and answer options
for i, question in enumerate(st.session_state.quiz_questions):
    st.subheader(f"Question {i + 1}")
    st.text(question)
    choice = st.radio(f"Choose your answer for question {i + 1}", ["A", "B", "C", "D"], index=None,
                      key=f"question_{i}")
    st.session_state.user_answers[i] = choice
    st.session_state.submit_disabled = False


def check_answers():
    # Validation
    for choice in st.session_state.user_answers:
        # If no Answer chosen
        if not choice:
            st.write(":red[In order to submit, all of the questions require an answer.]")
            return
    num_correct = sum(user_answer == correct_answer for user_answer, correct_answer in
                      zip(st.session_state.user_answers, st.session_state.correct_answers))
    total_questions = len(st.session_state.quiz_questions)
    st.write(f"Your score: {num_correct} out of {total_questions}")

    # Display correct answers
    for i, (question, correct_answer) in enumerate(
            zip(st.session_state.quiz_questions, st.session_state.correct_answers), 1):
        st.write(f"Question {i}: {question}")
        st.write(f"Correct answer: {correct_answer}")


# Submit quiz button
if st.button("Submit Quiz", disabled=st.session_state.submit_disabled):
    check_answers()




