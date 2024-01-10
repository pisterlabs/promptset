from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os
#required libraries are imported
os.environ["OPENAI_API_KEY"] = ""#Removed the api key. Please add the api key
#i have added the key to the openai_api_key environment variable
def create_the_prompt_template():
	template="""
	You are an expert quiz maker for technical topics.
	Create a quiz with 5{type} of questions about the following:{topic}
	For example if the topic is Data Structures and Algorithms and you have to generate programming questions:
	You can give following questions: Implement linked list, Implement bfs, solve the knapsack problems
	If you have to generate subjective questions on the same topic. You can give following questions: Write down the 
	time complexity of heap sort, bubble sort, bellman ford etc.
	If you have to generate multiple choice questions, you can give following questions: 
	What is the time complexity of heap sort?
	a)O(nlogn)
	b)O(n)
	c)O(1)
	d)O(n^2)
	"""
	prompt = PromptTemplate.from_template(template)
	return prompt 
#I have given the prompt and some examples to specify the type of questions
def create_quiz_chain(prompt_template,llm):
	return LLMChain(llm=llm, prompt=prompt_template)
#returns the quiz
def main():
	st.title("Quiz Generator")
	st.write("Write something about a topic and this generates a quiz")
	prompt_template = create_the_prompt_template()
	llm = ChatOpenAI()
	chain = create_quiz_chain(prompt_template,llm)
	topic = st.text_area("Enter something about the topic")
	quiz_type = st.selectbox("Select the type of questions",["multiple-choice","subjective","programming"])
	generate_button_clicked = st.button("Generate")
	if generate_button_clicked:
		quiz = chain.run(type=quiz_type,topic=topic)
		st.write(quiz)

if __name__=="__main__":
	main()
