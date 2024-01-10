import streamlit as st
from PyPDF2 import PdfFileReader as PdfReader
import openai
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from transformers import pipeline


st.markdown("""
# ✍️ AI-Powered Motivation Letter Generator
            
Generate a cover letter. All you need to do is:
1. Upload your resume or copy your resume/experiences
2. Paste a relevant job description
3. Enter your name, the company name, the hiring manager, the job title, and why you are motivated to work in this role
            
""")    


st.title("Please paste your own API key below")
API_Key = st.text_input("API Key")
if API_Key:
    openai.api_key= API_Key
    st.success("API Key set successfully")
    st.write(f"API key is: {API_Key}")
else:
    st.warning("Please read the instrucation for how to get the API key. https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt")
            
#input resume and job description
res_file = st.file_uploader('🗂️ Upload your resume in pdf format')
if res_file:
    pdf_reader = PdfReader(res_file)
    # Collect text from pdf
    res_text = ""
    for page in pdf_reader.pages:
        res_text += page.extractText()
    print(res_text)

    # other inputs

job_desc = st.text_input('Job description')  
user_name = st.text_input('Your name')
company = st.text_input('Company name')
manager = st.text_input('Hiring manager')
role = st.text_input('What role are you applying for?')
experience = st.text_input('I have experience in this role')
motivation = st.text_input('I am motivated to work in this role because....')
referral = st.selectbox("How did you find out about this opportunity?", ("LinkedIn", "Indeed", "Glassdoor", "Facebook","Other"))
ai_temp = st.number_input('AI Temperature (0.0-1.0) Input how creative the API can be',value=.7)

# submit button 
submitted = st.button("Generate Motivation Letter")
if submitted:

    # note that the ChatCompletion is used as it was found to be more effective to produce good results
    # using just Completion often resulted in exceeding token limits
    # according to https://platform.openai.com/docs/models/gpt-3-5
    # Our most capable and cost effective model in the GPT-3.5 family is gpt-3.5-turbo which has been optimized for chat 
    # but works well for traditional completions tasks as well.
  
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k", 
      #model ="La"
      #model = "gpt-3.5-turbo",
      #model="LaMDA",
      temperature=ai_temp,
      #my_key = API_Key,
      messages = [
            {"role": "user", "content" : f""" 
        In the second paragraph focus on why the candidate is a great fit drawing parallels between the experience included in the resume 
        and the qualifications on the job description.
        """},
                {"role": "user", "content" : f""" 
        In the 3RD PARAGRAPH: Conclusion
        Restate your interest in the organization and/or job and summarize what you have to offer and thank the reader for their time and consideration.
        """},
        {"role": "user", "content" : f""" 
        note that contact information may be found in the included resume text and use and/or summarize specific resume context for the letter
            """},
        {"role": "user", "content" : f"Use {user_name} as the candidate"},
        
        {"role": "user", "content" : f"Generate a specific cover letter based on the above. Generate the response and include appropriate spacing between the paragraph text"}
      ]
    )
            Contact at sujan.007.ice@gmail.com
    response_out = completion['choices'][0]['message']['content']
    st.write(response_out)
Contact at sujan.007.ice@gmail.com
    # include an option to download a txt file
    st.download_button('Download the cover_letter', response_out)
