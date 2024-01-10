import openai
openai.api_key = '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'
from langchain import PromptTemplate
import streamlit as st


def create_the_quiz_prompt_template():
    """Create the prompt template for the quiz app."""
    
    template = """
You are an expert quiz maker for technical fields. Let's think step by step and
create an MCQ quiz with {num_questions} questions about the following concept/content: {quiz_context}.

The format of the quiz is the following:
- Multiple-choice: 
- Questions:
    <Question1>: 
    <a. Answer 1>, 
    <b. Answer 2>, 
    <c. Answer 3>,
     <d. Answer 4>
    <Question2>: 
    <a. Answer 1>, 
    <b. Answer 2>, 
    <c. Answer 3>, 
    <d. Answer 4>
    ....
- Answers:
    <Answer1>: <a|b|c|d>
    <Answer2>: <a|b|c|d>
    ....
    Example:
    - Questions:
    What does the SQL acronym "DDL" stand for?

    a. Data Definition Language

    b. Data Description Language

    c. Data Design Language

    d. Database Design Language
    - Answers: 
        1. a
"""
    prompt = PromptTemplate.from_template(template)
    prompt.format(num_questions=3, quiz_context="Data Structures in Python Programming")
    
    return prompt

def quiz_gen_without_chain(prompt_question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful research and\
            programming assistant"},
                  {"role": "user", "content": prompt_question}]
    )
    
    return response["choices"][0]["message"]["content"]



def main():
    st.title("Quiz App")
    st.write("This app generates a quiz based on a given context.")
    prompt_template = create_the_quiz_prompt_template()
    context = st.text_area("Enter the concept/context for the quiz")
    if st.button("Generate Quiz"):
        quiz_response = quiz_gen_without_chain(context)
        st.write("Quiz Generated!")        
        st.write(quiz_response)
    if st.button("Show Answers"):
        st.write("-Scroll to above mentioned answers for self evalutaion-")

if __name__=="__main__":
    main()

