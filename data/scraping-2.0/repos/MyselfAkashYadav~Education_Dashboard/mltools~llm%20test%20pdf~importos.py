import streamlit as st
import openai


openai.api_key = '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'

def analyze_answer(question, user_answer):
    prompt = f"You are a bot who just has to analyze answers given by users for a given defined question. Give the score of the answer and review it. And tell where the user can make improvements.\n\nExample-\n\nQuestion:\nWhat is linked List?\nUser Answer:\nLinked list is a linear data structure.\n\nResponse:\nscore:\n4/10\nReview:\nThe answer is too short. A linked list is indeed a data structure. There is no need to correct this answer; it is accurate. A linked list is a linear data structure that consists of a sequence of elements, each of which contains a reference (link) to the next element in the sequence, forming a chain-like structure.\n\n\nQuestion:\n{question}\nUser Answer:\n{user_answer}"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,  
        stop=None  
    )


    return response.choices[0].text.strip()


st.title("Answer Review App")


question = st.text_input("Enter the Question:")
user_answer = st.text_area("Enter the User's Answer:")


if st.button("Analyze"):
    if not question or not user_answer:
        st.warning("Please provide both the question and the user's answer.")
    else:
        generated_response = analyze_answer(question, user_answer)

        st.subheader("Generated Review:")
        st.write(generated_response)
