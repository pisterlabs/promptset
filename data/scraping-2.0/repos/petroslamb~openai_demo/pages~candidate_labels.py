import json
import streamlit as st
import openai

st.markdown("# Candidate Labels 🎉")
st.sidebar.markdown("# Candidate Labels 🎉")

model_list = openai.Model.list()
model_names = [model["id"] for model in model_list["data"]]

model_names = ["text-davinci-003"] + model_names

model = st.sidebar.selectbox("Select a model", model_names, index=0)

# Get the model for the selected model
st.sidebar.write(openai.Model.retrieve(model))

@st.cache
def call_model(prompt):
    response = openai.Completion.create(
    model=model,
    prompt=prompt,
    temperature=0.7,
    max_tokens=756,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return json.loads(response["choices"][0]["text"])

job_description = '''
Python Software Engineer

We are looking for a highly skilled software engineer with experience in Python and cloud technologies. 
The successful candidate will be responsible for designing, implementing, and maintaining complex software systems, as well as collaborating with cross-functional teams to drive innovation and excellence.

Candidate Requirements:
- Bachelor's degree in Computer Science or a related field
- 5+ years of experience in software engineering
- Strong proficiency in Python and cloud technologies (e.g. AWS, Azure, GCP)
- Experience with agile development processes and best practices
- Javascript and React experience is a plus'''

resume = '''
John Doe
Software Engineer

Education:
- Bachelor's degree in Computer Science, XYZ University (2014-2018)

Experience:
- Software Engineer, ABC Corporation (2018-present)
- Developed and maintained Python-based software systems using AWS, Azure, and GCP
- Collaborated with cross-functional teams to drive innovation and excellence
- Utilized agile development processes and best practices
- Waiter at a restaurant (2014-2018)

Skills:
- Python
- Go (Golang)
- Cloud technologies (AWS, Azure, GCP)
- Agile development processes
- Best practices in software engineering
- Creating carpets
- Playing the flute

Contact:
- Email: john@example.com
- Phone: 555-555-5555'''

text_input = st.text_area("Enter a job description here", height=400, value=job_description)
text_input2 = st.text_area("Enter a resume here", height=400, value=resume)

button = st.button("Get Labels")

prompt ='''
I will provide you with a job description and a candidate resume.
You must match the candidate to the job description.
To do so create a list of matching keywords (labels) and their short descriptions explaing the label.
Return your response encoded in a json string, like so:

{"education": {"mscs": "Master\'s degree in Computer Science"}, "skills": {"golang": "Go (Golang)", "aws": "AWS"}, "responsibilities": {"team_lead": "Team Lead"}, "experience": {"10years": "10+ years of experience"}}

The job description and the resume are below:
''' + "\n\nJob Description:\n\n" + text_input + "\n\nResume:\n\n" + text_input2 + "\n\nLabels:\n\n",

if button:
    labels = call_model(prompt)
    st.write(labels)
