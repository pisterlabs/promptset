import openai
from openai import OpenAI
import streamlit as st
import pandas as pd
 
st.header("Content generation / Essay Writing")

#usr_title = st.text_input("Required output information","Company name, Head quarter, Established Yr, No. of Employees,Turnover INR, Market share, Market segment, Strength, Weakness,Latest info ")

with st.sidebar:
  usr_title = st.text_input("Title : ","Financial Literacy for Board of Directors")
  inp_context = st.radio("",["Indian context","Global context"],horizontal = True)
  inp_level = st.radio("",["Detailed","Simple"],horizontal = True)
  st.header("Sub titles")
  sub_heading = ["Project objective and Goal", "Introduction and Context", "Project Scope- how its implementation will help the organization", "The Project-specific details of the project and its implementation strategy", "Description of the deliverables, the timeframe and estimates of resources for execution", "Conclusion- Key findings & recommendations", "Acknowledgements", "Appendices and References"]
  prompt_inp = []
  for i in sub_heading:
    inp = st.text_input("",i,label_visibility="collapsed")
    prompt_inp.append(inp)

#   inp_sub_title1  = st.text_input(""," Project objective and Goal")
#   inp_sub_title2 = st.text_input("", "Introduction and Context")
#   inp_sub_title3 = st.text_input("","Project Scope- how its implementation will help the organization")
#   inp_sub_title4 = st.text_input("","The Project-specific details of the project and its implementation strategy")
#   inp_sub_title5 = st.text_input("","Description of the deliverables, the timeframe and estimates of resources for execution")
#   inp_sub_title6 = st.text_input("","Conclusion- Key findings & recommendations")
#   inp_sub_title7 = st.text_input("","Acknowledgements")
#   inp_sub_title8 = st.text_input("","Appendices and References")
	

# usr_input = usr_title+" in "+inp_context +" "+inp_level+" "+inp_sub_title1
# usr_input1 = usr_title+" in "+inp_context +" "+inp_level+" "+inp_sub_title2
# usr_input2 = usr_title+" in "+inp_context +" "+inp_level+" "+inp_sub_title3
# words = 2500
# if inp_level == "Detailed":
#   words = 5000
# usr_input = "I need a "+ inp_level + " content for " + usr_title + " in the " + inp_context + " with subheadings " + ", ".join(prompt_inp) + ". Each subheading should contain atleast 300 words"
# st.write(usr_input)

client = OpenAI(api_key = "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")
context = "I am preparing a dissertation on "+usr_title
# usr_input= "find the top 5 manufacturing companies from the provided context.show the results in table form"

gen_content = []

for i in range(len(prompt_inp)):
  usr_input = "Give me detailed, atleast 300 words of content for " + usr_title + " under the subheading " + prompt_inp[i] + " under the " + inp_context + ". Don't include the conclusion"
  completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
      {"role": "system", "content": context},
      {"role": "user", "content": usr_input},
  ]
  )
  st.header(prompt_inp[i])
  st.write(completion.choices[0].message.content)
  gen_content.append(completion.choices[0].message.content)

# final_inp = " ".join(gen_content) + "Rephrase this content"

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#       {"role": "system", "content": context},
#       {"role": "user", "content": final_inp},
#   ]
#   )
# st.header(final_inp)
# st.write(completion.choices[0].message.content)

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": context},
#    # {"role": "user", "content": usr_input},
#     {"role": "user", "content": usr_input}
#   ]
# )
# st.write("Result")
# st.header(inp_sub_title2)
# st.write(completion.choices[0].message.content)

#st.header(inp_sub_title2)
#st.write(completion.choices[1].message.content)