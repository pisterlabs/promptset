import os
import openai

class StudentsInfoSummarize:
    def __init__(self, students_info):
        self.students_info = students_info

    def add_student(self, name, age, grade):
        print("add_student")

    # pass one student info string to summarize_one_student
    def summarize_one_student(self, student_info):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.ChatCompletion.create(
        model="gpt-3.5",
        messages=[
            {
            "role": "system",
            "content": "Despite the similarity in some students' names, it's crucial to remember that they are unique individuals. Please provide separate summaries for each student's background. Include their name, Uci Net and background information."
            },
            {
            "role": "user",
            "content": student_info
        },
        ],
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
       )
        print(response);
