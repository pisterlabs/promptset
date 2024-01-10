import openai
import shutil
import datetime
import os
from common import load_file_text, write_file

openai.api_key = variable = os.environ.get('OPENAI_API_KEY')



messages="""
read my experiance below.
\n
""" + load_file_text("experiance.txt") + "\n" + """
--------------------------------------------------------------\n
following is the job posting I am trying to apply. \n
""" + load_file_text("inputs/1-job.txt") + "\n" + """
--------------------------------------------------------------\n
write me three cover letters for this job posting. 
- if you can find the name of the hiring manager, please use it, if not address the team at the company.
- do not mention C# unless it is requred in the job posting.
- do not mention React unless it is requred in the job posting.
- cover letter will be submitted via a web form.
- split each your coverletter for a letter by a '<---------------------------------->'

IMPORTENT: Makesure you do not lie about my skills. Unless I don't have mentioend in my experiance, do not say I have it.
"""

file = 'output/1-prompt-for-cover-letters.txt'
write_file(file,messages)

print( 'asking for cover letters. file printed to ' + file)

response = openai.ChatCompletion.create(
    model="gpt-4",
    max_tokens=5000,
    temperature=1,
    messages = [{'role':'user', 'content':load_file_text('output/1-prompt-for-cover-letters.txt')}]
)


# print(response)
print( 'asking for cover letters. response printed to inputs/3-cover-letters.txt')
write_file('inputs/3-cover-letters.txt', response.choices[0].message.content)
write_file('output/0-cover-letters.txt', response.choices[0].message.content)
cover_letters = response.choices[0].message.content


prompt = """
your name is Jan and you are a hiring manager. 
You are hiring for the following position. """ + load_file_text("inputs/1-job.txt") + """
\n------------------------------\n
evaluate following cover letters for the position.
give a score and justification. split each your justification for a letter by a '<---------------------------------->'  \n\n """ + load_file_text("inputs/3-cover-letters.txt")

filename = 'output/2-prompt-for-scoring.txt'
write_file(filename, prompt)

print( 'asking for cover letters scoring. file printed to ' + filename)

response = openai.ChatCompletion.create(
    model="gpt-4",
    max_tokens=4000,
    temperature=1,
    messages = [{'role':'user', 'content':load_file_text('output/2-prompt-for-scoring.txt')}]
)
# print(response)

print( 'got answer for cover letters scoring. file printed to output/3-cover-scores.txt')
write_file('output/3-cover-scores.txt', response.choices[0].message.content)
score_response = response.choices[0].message.content


# print(response)

print('got answer for cover letters scoring. file printed to output/3-cover-scores.txt')
write_file('output/3-cover-scores.txt', response.choices[0].message.content)
score_response = response.choices[0].message.content

cover_letters = cover_letters.split('----------------------------------')
score_response = score_response.split('----------------------------------')

# create a new file and write the contents to it
with open('output/4-cover-letters-with-scores.html', 'w') as file:
    for i in range(len(cover_letters)):
        file.write(f'<h2>Cover Letter {i+1}:</h2>')
        file.write(f'<p>Score: {score_response[i]}</p>')
        file.write(f'<p>Justification: {cover_letters[i]}</p>')
        file.write('<hr>')

print('Cover letters with scores saved to output/4-cover-letters-with-scores.html')

# prompt the user for a job name
job_name = input('Enter a job name (leave blank for timestamp): ').strip()

# use the timestamp as the default answer if the user input is empty
if not job_name:
    job_name = datetime.datetime.now().strftime('%y%m%d%H%M%S')

shutil.copy('output/4-cover-letters-with-scores.html', f'output/htmls/{job_name}.html')

print(f'Cover letters with scores saved to output/htmls/{job_name}.html')


# open the HTML file in the default web browser
import webbrowser
import os
webbrowser.open(f'file://{os.path.abspath(f"output/htmls/{job_name}.html")}', new=2)
