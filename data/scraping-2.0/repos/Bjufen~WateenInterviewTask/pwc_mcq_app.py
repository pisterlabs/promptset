import os
from API_key import apikey
import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey


@st.cache_data
def generate_question(prompt, number):
    if not prompt:
        return []

    question_template = PromptTemplate(
        input_variables=['topic', 'number'],
        template='Create exactly {number} multiple-choice questions about {topic}. Please only give 4 possible responses.'
                 'I want the response for each question to be in the following template:'
                 'Question [NUMBER]: [QUESTION]'
                 'Option A: [OPTION A]'
                 'Option B: [OPTION B]'
                 'Option C: [OPTION C]'
                 'Option D: [OPTION D]'
                 'Correct Answer: [A, B, C OR D]'
    )

    questions = []

    llm = OpenAI(temperature=0.9)
    question_chain = LLMChain(llm=llm, prompt=question_template, verbose=True)

    response = question_chain.run(topic=prompt, number=number)

    question_blocks = response.split("Question ")[1:]

    for block in question_blocks:
        # Split the block into lines
        lines = block.strip().split('\n')

        # Extract question
        question = lines[0][2:].strip()

        # Extract possible answers
        possible_answers = [line[9:].strip() for line in lines[1:5]]

        # Extract correct answer and convert to integer index
        correct_answer_text = lines[5][len("Correct Answer: "):].strip()
        correct_answer_index = ord(correct_answer_text) - ord('A')

        # Create QuizQuestion object and append to the list
        quiz_question = QuizQuestion(question, possible_answers, correct_answer_index)
        questions.append(quiz_question)

    return questions


def main():
    st.title('🦜️ MCQ Quiz Application')

    # Get user input for the quiz topic
    quiz_topic = st.text_input('Enter the quiz topic: ')

    # Get user input for the number of questions
    num_questions = st.number_input('Enter the number of questions:', min_value=1, step=1, max_value=5)

    # Initialize variables to store user's answers
    user_answers = []

    # Generate and display questions
    questions = generate_question(quiz_topic, num_questions)

    for i, question in enumerate(questions):
        st.write(f"\n\nQuestion {i + 1}: " + question.question)
        selected_option = st.radio('Choose an option: ', question.possible_answers, key=i)
        user_answers.append((question, selected_option))

    # Submit button to check answers
    if st.button('Submit'):
        # Check and display results
        correct_answers = 0
        for user_answer in user_answers:
            if user_answer[1] == user_answer[0].possible_answers[user_answer[0].correct_answer]:
                correct_answers += 1

        # Display user's answers and result
        st.write("\n\nYour Answers:")
        for question, user_answer in user_answers:
            st.write(f"Q: {question.question} - Your Answer: {user_answer}")

        # Display quiz results
        st.write("\n\nQuiz Results:")
        st.write(f"Total Questions: {num_questions}")
        st.write(f"Correct Answers: {correct_answers}")


class QuizQuestion:
    def __init__(self, question, possible_answers, correct_answer):
        self.question = question
        self.possible_answers = possible_answers
        self.correct_answer = correct_answer


if __name__ == '__main__':
    main()
