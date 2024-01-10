
import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
import openai

load_dotenv()

#os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
openai.api_key = "YOUR_OPENAI_KEY"



yaml_content = """
models:
- type: main
  engine: openai
  model: text-davinci-003
"""


from nemoguardrails import LLMRails, RailsConfig

# initialize rails config
f = open('co_shop.txt','r')
contents = f.read()

f = open('co_profanity.txt','r')
contents_1= f.read()

f = open('co_company.txt','r')
contents_2= f.read()

f = open('co_hateSpeech.txt','r')
contents_3= f.read()


Selected_colangs = ""
Li_SelectedRules = []  #list of selected rules, used this in save config
##############################################################################
Politics = st.checkbox('Guardrails to prevent Politics relevant Query')
Profanity = st.checkbox('Guardrails for Profanity Filter')
Company = st.checkbox('Guardrails for Company confidential Information Filter')
Hate = st.checkbox('Guardrails for Filter-out Hate Speech')
if Politics:
    Selected_colangs = Selected_colangs + contents
    Li_SelectedRules.append(contents)
if Profanity:
    Selected_colangs = Selected_colangs + contents_1
    Li_SelectedRules.append(contents_1)
if Company:
    Selected_colangs = Selected_colangs + contents_2
    Li_SelectedRules.append(contents_2)
if Hate:
    Selected_colangs = Selected_colangs + contents_3
    Li_SelectedRules.append(contents_3)
else:
    pass
##############################################################################

config = RailsConfig.from_content(
  	yaml_content=yaml_content,
    colang_content=Selected_colangs
)
# create rail
rails = LLMRails(config)



def without_guardrails(text):
    response = openai.Completion.create(
        prompt="Enter Your Query\n"+text,
        engine="text-davinci-003",
        max_tokens=2048,
        temperature=0)

    result = response['choices'][0]['text']
    return result

def with_guardrails(text):
    res = asyncio.run(rails.generate_async(prompt=text))
    return res

def main():
    st.title("Guardrails Demo")
    text_area = st.text_area("Enter Query")

    if st.button("Submit Query"):
        if len(text_area) > 0:

            st.warning("Output Without Guardrails")

            without_guardrails_result = without_guardrails(text_area)
            st.success(without_guardrails_result)

            st.warning("Output With Guardrails")


            with_guardrails_result = with_guardrails(text_area)
            st.success(with_guardrails_result)


if __name__ == '__main__':
    main()
