import streamlit as st
import cohere
import os


api_key = os.environ.get('CO_API_KEY')
co = cohere.Client(api_key)


def generate_mcq(text,number):
    """Generate multiple choice questions given the text and the number of questions. 
    Arguments: text
    number (int)
    Returns: MCQ:
    """
    prompt = f""" Given a TEXT, generate the specified NUMBER of multiple choice questions based on the TEXT. Provide 4 options for each multiple choice question. Ensure there is only one correct answer for each multiple choice question. Provide the answers to the questions at the end of the response with a brief explanation as to why the correct option is correct and the rest of the options are wrong. ".
    TEXT: {text}
    NUMBER: {number}
    """



    response = co.generate(
          model="command-nightly",
        prompt=prompt,
        max_tokens=5000,
        temperature=0.7,
        k=0,
    )
    MCQ = response.generations[0].text
    MCQ = MCQ.replace("\n\n--", "").replace("\n--", "").strip()
    MCQ = MCQ.split("\n")

    return MCQ






def run():
    st.title("✨MCQ GENERATOR ✨")
    text = st.text_area("Paste the text that you would like to generate multiple choice questions for:")
    number = st.number_input("Input the number of multiple choice questions to be generated:", min_value =0, value = 0, step = 1)
    
    if st.button("Generate MCQ"):
        if text and number: 
            MCQ = generate_mcq(text,number)
            for i in MCQ: 
             st.write(i)

if __name__ == "__main__":
    run()
    

