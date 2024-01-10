import openai 
from dotenv import load_dotenv

load_dotenv()

prompt = """Calculate the similarity score between the following text and the job description provided:

 

Text:
0     Python, C, C++, Numpy, Pandas, Matplotlib, Se...
Name: Technical Skills, dtype: object

 

Job Description:
Proven experience as a Machine Learning Engineer or similar role
Understanding of data structures, data modeling and software architecture
Deep knowledge of math, probability, statistics and algorithms
Ability to write robust code in Python, Java and R
Familiarity with machine learning frameworks (like Keras or PyTorch) and libraries (like scikit-learn)
Excellent communication skills
Ability to work in a team
Outstanding analytical and problem-solving skills
BSc in Computer Science, Mathematics or similar field; Master’s degree is a plus

 
response as a python syntax float value between 0 and 1.
response: 
"""

response = openai.ChatCompletion.create(
model="gpt-3.5-turbo",
messages=[
            {"role": "system", "content": "You are a meditation expert."},
            {"role": "user", "content": prompt},
        ],
max_tokens=700,
temperature=0.7
)