import openai
from openai import OpenAI
import streamlit as st
from basecode.authenticate import return_api_key
import os
import pandas as pd

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=return_api_key(),
)

def call_api():
	prompt_design = st.text_input("Enter your the prompt design for the API call:", value="You are a helpful assistant.")
	prompt_query = st.text_input("Enter your prompt query:", value="Tell me about Singapore in the 1970s in 50 words.")
	if st.button("Call the API"):
		if prompt_design and prompt_query:
			api_call(prompt_design, prompt_query)
		else:
			st.warning("Please enter a prompt design and prompt query.")
	

def api_call(p_design, p_query):
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
	st.title("Api Call")
	MODEL = "gpt-3.5-turbo"
	with st.status("Calling the OpenAI API..."):
		response = client.chat.completions.create(
			model=MODEL,
			messages=[
				{"role": "system", "content": p_design},
				{"role": "user", "content": p_query},
			],
			temperature=0,
		)

		st.markdown("**This is the raw response:**") 
		st.write(response)
		st.markdown("**This is the extracted response:**")
		st.write(response.choices[0].message.content)
		completion_tokens = response.usage.completion_tokens
		prompt_tokens = response.usage.prompt_tokens
		total_tokens = response.usage.total_tokens

		st.write(f"Completion Tokens: {completion_tokens}")
		st.write(f"Prompt Tokens: {prompt_tokens}")
		st.write(f"Total Tokens: {total_tokens}")

def rule_based():
	st.write("Rules for the chatbot:")
	df = pd.DataFrame(
		[
			{"prompt": "Hello", "response": "Hi there what can I do for you"},
			{
				"prompt": "What is your name?",
				"response": "My name is EAI , an electronic artificial being"
			},
			{"prompt": "How old are you?", "response": "Today is my birthday!"},
		]
	)

	edited_df = st.data_editor(df, num_rows="dynamic")
	st.divider()
	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("Enter your prompt"):
		if prompt in edited_df["prompt"].values:
			reply = edited_df.loc[edited_df["prompt"] == prompt]["response"].values[0]
		else:
			reply = "I don't understand"

		with st.chat_message("user"):
			st.write(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("assistant"):
			st.write(reply)
			st.session_state.messages.append({"role": "assistant", "content": reply})