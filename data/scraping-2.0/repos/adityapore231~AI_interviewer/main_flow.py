import streamlit as st
import requests
import openai
import PyPDF2
#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

openai.api_key = "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"
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text().strip()
        return text
    except Exception as e:
        st.error(f'Error extracting text from PDF: {e}')
        return None
def get_openai_response(prompt, model='gpt-3.5-turbo'):
  pass
def app():
# Set page title and icon
  st.set_page_config(page_title='AI Interviewer Assistant', page_icon=':robot_face:')

  st.write('## AI Interviewer Assistant')
  st.write('##### Fill the below information to start generating questions for your candidate.')

# Initial Prompt
  uploaded_file = st.file_uploader('''Upload candidate's resume''')
  job_role = st.text_input('Job Role/Position')
  job_description = st.text_input('Job Description')
  difficulty_level = st.slider('Select Difficulty Level', min_value=1, max_value=5, step=1, value=3)
  years_of_experience = st.selectbox('Years of Experience', ['0-2', '2-5', '5+'])
  num_questions = st.number_input('How many questions?', min_value=1, max_value=10)
  text = extract_text_from_pdf(uploaded_file)
  role_prompt = f"You are interviewer today. You are taking interview for the position {job_role}. The job description is {job_description}. you are hiring for {years_of_experience} year experience people. difficulty level will be {difficulty_level} out of 5. Ask {num_questions} question related to project / work experience from resume first."
  print(role_prompt)
  if st.button('Generate'):
    completions = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": f"The candidate resume is this {text}."},
      ]
    )
    response = completions['choices'][0]['message']['content']

    st.write(response.strip())
    
    #response = get_openai_response(initial_prompt)
    #st.write(f"Following are the {num_questions} that you can ask to candidate:")
    #st.write(initial_prompt)
    #ask_question = st.text_input("Ask specific qeustion e.g Projects")
    #if st.button('Ask'):
       #response1 = get_openai_response(ask_question)
       #st.write(response1)


if __name__ == "__main__":
  app()
