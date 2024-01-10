"""
This class is used to generate an article canvas from a list of questions.
"""

import os
import openai
import json

class Canvas:
    def __init__(self):
        pass

    def create_canvas(self, title, input_file, output_file, options = 3):
        questions = self.read_input_file(input_file)
        canvas = self.create_article_canvas(title, questions, options)
        self.save_content(canvas, output_file)
    
    def generate_markdown(self, canvas):
        pass
        
    
    def add_new_answer(self, json_file_name, question_number):
        canvas = self.read_json_file(json_file_name)
        question = canvas["questions"][question_number]
        answers = question["answers"]
        new_answer = self.get_answser(question["question"])
        answers.append(new_answer)
        canvas["questions"][question_number]["answers"] = answers
        self.save_content(canvas, json_file_name)
        print(f"New answer added to question {question_number}")
        print(f"New answer: {new_answer}")
    
    def create_article_canvas(self, title, questions, options):
        canvas = {
            "title": title,
            "introduction": self.generate_introduction(title),
            "summary": self.generate_summarty(title),
            "questions": {}
        }
        for i, question in enumerate(questions):
            answers = self.add_answers(question, options)
            canvas["questions"][f"q{i}"] = {
                "type": "text",
                "question": question,
                "answers": answers,
            }
        return canvas
    
    def generate_introduction(self, title):
        return self.get_answser(f"Generate a blog post intro about the following topic {title}")
    
    def generate_summarty(self, title):
        return self.get_answser(f"Generate a blog post summary about the following topic {title}")
    
    def read_input_file(self, input_file):
        with open(input_file, 'r') as f:
            questions = f.readlines()
        return self.remove_newline_characters(questions)
    
    def remove_newline_characters(self, questions):
        return [question.strip() for question in questions]
    
    def show_canvas(self, json_file_name):
        self.print_content(json_file_name)
    
    def read_json_file(self, json_file_name):
        with open(json_file_name, 'r') as f:
            return json.load(f)
    
    def print_content(self, json_file_name):
        article_canvas = self.read_json_file(json_file_name)
        print(json.dumps(article_canvas, indent=4))

    def add_answers(self, question, options):
        answers = []
        print(f"Generating an answer for a question {question}", end =" ")
        for i in range(options):
            answer = self.get_answser(question)
            answers.append(answer)
            print(".", end =" ", flush=True)
        print("Done.")
        return answers
    
    def get_answser(self, question, max_tokens=1000, temperature=1):
        openai.api_key = os.environ['OPENAI_API_KEY']
        response = openai.Completion.create(engine='text-davinci-002', 
                                            prompt=question,
                                            max_tokens=max_tokens,
                                            temperature=temperature)
        return response.choices[0].text
    
    def save_content(self, canvas, output_file):
        with open(output_file, 'w') as f:
            f.write(json.dumps(canvas, indent=4))