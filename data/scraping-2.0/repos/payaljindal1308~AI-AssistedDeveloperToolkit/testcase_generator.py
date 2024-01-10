
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
llm = OpenAI(temperature=0.5, max_tokens=1000)

def generate_test_cases(user_input):
    template = """
    You are a helpful assistant that can generate SQL queries based on user input.

    User Input: {user_input}
    """
    test_case_template = PromptTemplate(
    input_variables=['user_input'],
    template="""You're using a Test Case Generator.
    Write a test case in python to validate the correctness of the code snippet or function- {user_input}. Please provide the whole code"""
    )

    cases_memory = ConversationBufferMemory(input_key='user_input', memory_key='chat_history')
    
    # LLM
    testcase_chain = LLMChain(llm=llm, prompt=test_case_template, verbose=True, output_key='script', memory=cases_memory)

    generated_code = testcase_chain.run(user_input)
    return generated_code

# App framework
st.title('Test Case Generator')
user_input = st.text_area('Code Snippet or Function')

if st.button('Generate Test Cases'):
    # Generate code using LLM
    generated_cases = generate_test_cases(user_input=user_input)
    
    generated_cases_lines = generated_cases.split('\n')
    generated_cases_lines = generated_cases_lines[1:]  # Remove the first line
    generated_cases = '\n'.join(generated_cases_lines)
    # Write the test cases to a Python file
    with open('test_cases.py', 'w') as file:
        file.write(generated_cases)

    # Display the generated test cases
    print(generate_test_cases)
    st.write('Generated Test Cases:')
    st.text_area('generated_cases', value=generated_cases, height=300)
    st.success('Test cases generated successfully!')
