# Import necessary libraries
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# OpenAI API key for authentication
api = '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'

# Function to call OpenAI and generate quiz questions
def callOpenAi(topic, numOfQ):
    # Define the expected response schema for OpenAI output
    response_schema = ResponseSchema(
        name="questions", 
        subschemas=[
            ResponseSchema(
                name="question", 
                description="The text of the question",
                type="str"
            ),
            ResponseSchema( 
                name="options",
                description="Multiple choice options",
                type="str",
                subschemas=[
                    ResponseSchema(name="option", description="A multiple choice option", type="str")
                ]
            ),
            ResponseSchema(
                name="answer", 
                description="The correct answer option",
                type="str"
            )
        ],
        description="The list of questions",
        type="list of dict"
    )

    # Initialize output parser based on the response schema
    output_parser = StructuredOutputParser.from_response_schemas([response_schema])  
    format_instructions = output_parser.get_format_instructions()

    # Define the prompt template for OpenAI
    prompt = """
    You are an MCQ quiz guru, you have to generate {num} mcq questions about {topic}.
    Provide the right answer for each question and return the response in JSON format. 
    Here is an example of how the JSON structure should be in this format {format_instructions}.
    All the values in the JSON should not be prefixed with anything like "A.", "A)", "A:", "A:-".
    """

    # Create a ChatPromptTemplate object from the template
    prompt = ChatPromptTemplate.from_template(prompt)
    
    # Initialize ChatOpenAI model with the OpenAI API key
    model = ChatOpenAI(openai_api_key=api)

    # Define the processing chain: prompt -> model -> output_parser
    chain = prompt | model | output_parser

    # Invoke the processing chain with the input parameters
    answer = chain.invoke({"num": numOfQ, "topic": topic, "format_instructions": format_instructions})
    return answer

# Streamlit application title
st.title("Question Generator")

# Input fields for topic and number of questions
topic = st.text_input("Enter the topic of the questions:")
num_questions = st.number_input("Enter the number of questions:", min_value=1)

# Button to start the quiz
start_quiz_button = st.button("Start Quiz")

# Check if the quiz has started
if start_quiz_button:
    if topic and num_questions > 0:
        # Check if quiz data is not stored in session state
        if not 'quiz_data' in st.session_state:
            # Retrieve the selected topic and number of questions
            selected_topic = topic
            selected_num_questions = num_questions

            # Call OpenAI to generate quiz questions
            quiz_JSON = callOpenAi(selected_topic, selected_num_questions)

            # Initialize question index for tracking progress
            question_index = 0

            # Store quiz data in session state
            st.session_state['quiz_data'] = {
                'selected_topic': selected_topic,
                'selected_num_questions': selected_num_questions,
                'questions': questions,
                'question_index': question_index,
            }
        else:
            # Subsequent interactions, use stored state
            selected_topic = st.session_state['quiz_data']['selected_topic']
            selected_num_questions = st.session_state['quiz_data']['selected_num_questions']
            questions = st.session_state['quiz_data']['questions']
            question_index = st.session_state['quiz_data']['question_index']

        # Check if 'questions' key exists in the JSON response
        if 'questions' in quiz_JSON:
            questions = quiz_JSON['questions']

            # Check if there are questions available
            if len(questions) > 0:
                # Get the current question
                current_question = questions[question_index]
                st.write(current_question['question'])

                # Display radio buttons for answer options
                user_input = st.radio(f"Select an answer for Question {question_index + 1}:", current_question['options'], key=f"question_{question_index}")

                # Check the answer on submit
                if st.button("Submit"):
                    if user_input == current_question['answer']:
                        result = "Success! That's the correct answer."
                    else:
                        result = f"Wrong answer. The correct answer is '{current_question['answer']}'."
                    st.write(result)

                    # Check if there are more questions
                    if question_index < len(questions) - 1:
                        question_index += 1
                        current_question = questions[question_index]
                        st.write(current_question['question'])
                        user_input = st.radio(f"Select an answer for Question {question_index + 1}:", current_question['options'], key=f"question_{question_index}")
            else:
                st.write("No questions found for this topic.")
        else:
            st.write("Invalid quiz data.")
    else:
        # Display errors for missing or invalid input
        if not topic:
            st.error("Please enter a valid topic.")

        if num_questions <= 0:
            st.error("Please enter a valid number of questions (minimum 1).")
