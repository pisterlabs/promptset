import requests
import openai
from tqdm import tqdm

# ChatGPT usage: Partial
openai.api_key = "API_KEY"

api_url = 'https://ubcgrades.com/api/v3/subjects/UBCV'
response = requests.get(api_url, timeout=10)

if response.status_code == 200:
    data = response.json()
    subjects = []

    for d in data:
        subjects.append(d['subject'])

    res_texts = []

    for subject in tqdm(subjects, desc="Getting courses for each subject"):
        print(f'Getting courses for {subject}')
        subject_url = f'https://ubcgrades.com/api/v3/courses/UBCV/{subject}'
        subject_res = requests.get(subject_url, timeout=10)

        if subject_res.status_code == 200:
            data = subject_res.json()
            courses = []
            for obj in data:
                courses.append((subject, obj["course"], obj["course_title"]))

            course_list_str = "\n".join([f"{subject}{course} - {course_title}" for subject, course, course_title in courses])

            conversation = [
                {"role": "user", "content": f"Categorize these courses into several specific groups \
                    based on their titles, content, and potential themes: {course_list_str}"}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation
            )

            response_text = response['choices'][0]['message']['content']
            res_texts.append(response_text)
            with open('courses.txt', 'a') as f:
                f.write("\n".join(response_text.split("\n")[1:]))
        else:
            print(f"API call failed with status code {subject_res.status_code}")
else:
    print(f"API call failed with status code {response.status_code}")