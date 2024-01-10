# Importing required packages
import streamlit as st
import openai
from sqlalchemy import create_engine, text
import pandas as pd
from pandas.api.types import is_numeric_dtype

engine = create_engine(st.secrets["quasar_readonly_connection_string"])
conn = engine.connect()

st.title("Quasar GPT")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''
This is a web application that allows you to interact with our quasar data warehouse using natural language. 

- Enter a **query** in the **text box** and **click Send** to receive 
       a **response**. 
- Note: **This is an experiment, this data is not necessarily correct** 😅 
- If you want a plot output, try to structure your query such that it lends itself to being plotted. For example, '# of signups last week' will logically output just one number, but '# of signups per day last week' would be a good bar chart candidate. Remember to click 'Try to Plot?' if you want it to attempt to plot the response. 
- Also, errors have not yet been handled, so please refresh if you get an error message. 
- If you're experiencing an error that prevents even the output of the SQL query, please try clicking 'Use GPT 3.5 instead of 4' as GPT-4 is often overloaded with other requests. 
	-	Note that this will decrease the quality of the response somewhat, as GPT 3.5 is not quite as powerful.
       '''
    )

openai.api_key = st.secrets["openai_api_key"]

with open('quasar_gpt_test_system_prompt.txt', 'r') as file:
	system_prompt = file.read()

def get_data(sql_query):
	return pd.read_sql_query(text(sql_query), conn)

def ChatGPT(user_query):
	if use_old_model: 
		model_to_use = 'gpt-3.5-turbo'
	else: 
		model_to_use = 'gpt-4'
	completion = openai.ChatCompletion.create(
	  model=model_to_use,
	  messages=[
	  	 {"role": "system", "content": system_prompt},
	    {"role": "user", "content": user_query}
	  ]
	)

	return completion.choices[0].message

def submit():
	if user_query != "":
		# Pass the query to the ChatGPT function
		with st.spinner(text="Processing..."):
			query_to_send = str(user_query)
			if try_to_plot:
				query_to_send = query_to_send + " - the resulting table should have two columns, 'x' and 'y', so that I can plot this as a line chart"
			response = ChatGPT(query_to_send).content
			sql_query = response
			if '```sql' in sql_query:
				sql_query = sql_query[sql_query.find('```sql') + len('```sql'):sql_query.rfind('```')]
			st.code(sql_query, language='sql')
			df_to_display = get_data(sql_query)
			st.download_button(label = "Download data as CSV", data=df_to_display.to_csv(index=False), file_name='quasar_gpt_output.csv',mime='text/csv')
		st.success("Done!")
		if 'x' in df_to_display and 'y' in df_to_display:
			if is_numeric_dtype(df_to_display['x']):
				st.line_chart(data=df_to_display, x='x',y='y')
			else:
				st.bar_chart(data=df_to_display, x='x',y='y')
		return st.dataframe(df_to_display)

	else:
		st.write("Please enter a question.")

user_query = st.text_input("Enter query here", placeholder="for example: # of signups last week")
try_to_plot = st.checkbox('Try to Plot?')
use_old_model =st.checkbox('Use GPT 3.5 instead of 4')
st.button('Send',on_click=submit)
