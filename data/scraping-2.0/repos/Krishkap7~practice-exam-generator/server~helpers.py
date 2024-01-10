import os
import subprocess

from openai import OpenAI
from fastapi import UploadFile
from PyPDF2 import PdfReader
from latex import build_pdf
from pathlib import Path


class ExamGenerator:
    def __init__(self):
        self.system_prompt: str = "You are now a practice exam generator. From now on, \
            you will generate practice exam's in LaTex code using data coming from a user. \
            The user will uploaded text from pdfs containing materials from a university level course. " \
                                  "From these materials, output a practice exam written in LaTex that tests " \
                                  "the materials given. Do not output any text, only output latex code."

        self.user_prompt: str = "Use the information below to generate a practice exam of at least \
                                 5 questions. If you need to use multiple messages to write the code \
                                 in the limit, end exactly when you have to. This means do not end the " \
                                "latex document, just simply away for another 'continue' message. " \
                                "Do not output anything except LaTex code."
        self.client = OpenAI(
            api_key = ''
        )

    def generate_exam(self, files: list[UploadFile]):
        pdf_text = ''
        for file in files:
            pdf_text += self.extract_text(file) + '\n'
        user_prompt = self.user_prompt + pdf_text
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}]
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        ).choices[0].message.content
        latex = response

        while 'end{document}' not in response:
            messages.append({'role': 'user', 'content': 'continue'})
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            ).choices[0].message.content
            latex += response

        with open('local_file.tex', 'w') as f:
            f.write(latex)

        subprocess.run(['pdflatex', './local_file.tex'], check=True)

        with open('local_file.pdf', 'rb') as f:
            return f

    @staticmethod
    def extract_text(pdf_file: UploadFile) -> str:
        text = ""
        for page in pdf_file.pages:
            text += page.extract_text()
        return text


if __name__ == '__main__':
    e = ExamGenerator()
    pdf_path = Path('./practice_midterm_combined.pdf')
    pdf = PdfReader(pdf_path)
    print(e.generate_exam([pdf]))
