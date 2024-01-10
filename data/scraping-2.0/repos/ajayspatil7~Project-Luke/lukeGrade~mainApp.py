# Required Library
import os
import interfaceBuilder as iB
import apiKey
import streamlit as stl
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apiKey.apiKey
model = OpenAI(temperature=0.9)

# (Heading) UI Component 1
iB.IB_HeadingsViewer()

# (Language Choice) UI Component 2
user_lang_choice = iB.IB_Languages_Radio()

# (User Input) UI Component 3
# user_prompt = stl.text_input('Paste Your Code Here')
# Text area,
user_prompt = stl.text_area("**Paste your code here**", height=400)
automatedTestCases = iB.IB_AutoTestCases()
test_cases = stl.text_area("**Paste your test cases here**", height=100)
stl.write(":red[May give wrong answers sometimes]")
omitOut = ""
if automatedTestCases:
    feedModel = iB.IB_LanguageModelQuery(user_lang_choice, user_prompt)
    omitOut = model(feedModel)
else:
    feedModel = iB.IB_LanguageModelQuery(user_lang_choice, user_prompt, test_cases)
    omitOut = model(feedModel)

if stl.button('Submit your code'):
    with stl.spinner("Submitting your code..."):
        stl.write(omitOut)
