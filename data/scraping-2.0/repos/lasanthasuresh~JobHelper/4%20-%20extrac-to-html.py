import openai
import shutil
import datetime
from common import load_file_text, write_file


cover_letters = load_file_text('output/0-cover-letters.txt')
score_response = load_file_text('output/3-cover-scores.txt')

cover_letters = cover_letters.split('<---------------------------------->')
score_response = score_response.split('<---------------------------------->')

# create a new file and write the contents to it
with open('output/4-cover-letters-with-scores.html', 'w') as file:
    for i in range(len(cover_letters)):
        file.write(f'<h2>Cover Letter {i+1}:</h2><div>')
        file.write(f'{cover_letters[i]}')
        file.write('</div><br><div>')
        file.write(f'{score_response[i]}')
        file.write('</div><hr>')
        file.write('<br>')

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
