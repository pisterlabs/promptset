# Contents of page2.py
import streamlit as st

def app():
    st.title('Basics of Langchain')
    
    st.header('What is Langchain?')
    st.write("Langchain is a powerful tool for working with large language models (LLMs) that simplifies the process of composing these pieces and provides an abstraction for building custom knowledge chatbots. It works by taking a large source of data, breaking it down into chunks, and embedding them into a Vector Store. When a prompt is inserted into the chatbot, Langchain queries the Vector Store for relevant information, which is then used in conjunction with the LLM to generate the answer. [freecodecamp.org](https://www.freecodecamp.org/news/langchain-how-to-create-custom-knowledge-chatbots/)")

    st.header('Why Do We Need Langchain?')
    st.write("Langchain offers a useful approach to overcome the limitations of LLMs by preprocessing the corpus of text, breaking it down into chunks or summaries, embedding them in a vector space, and searching for similar chunks when a question is asked. This pattern of preprocessing, real-time collecting, and interaction with the LLM is common and can be used in other scenarios, such as code and semantic search. Langchain provides an abstraction that simplifies the process of composing these pieces, making it easier to work with large language models. [medium.com](https://medium.com/databutton/getting-started-with-langchain-a-powerful-tool-for-working-with-large-language-models-286419ba0842)")

    st.header('Example: Building a Question-Answering App with Langchain')
    st.write("Let's build a simple question-answering app using Langchain. Here's a basic example of how you can use Langchain to achieve this: [kdnuggets.com](https://www.kdnuggets.com/2023/04/langchain-101-build-gptpowered-applications.html)")

    st.subheader('Step 1: Install Langchain')
    st.code("pip install langchain", language="bash")

    st.subheader('Step 2: Import required libraries')
    st.code("""
import langchain as lc
from langchain import SimpleSequentialChain
    """, language="python")

    st.subheader('Step 3: Load a large language model')
    st.code("""
model = lc.load("gpt-3")
    """, language="python")

    st.subheader('Step 4: Define a function to answer questions')
    st.code("""
def get_answer(prompt):
    chain = SimpleSequentialChain(model)
    chain.add_prompt(prompt)
    response = chain.generate()
    return response
    """, language="python")

    st.subheader('Step 5: Get answers to your questions')
    st.code("""
question = "What is the capital of France?"
answer = get_answer(question)
print(answer)
    """, language="python")

    st.write("In this example, we used Langchain to build a simple question-answering app. You can further explore Langchain to build more interesting applications.")
