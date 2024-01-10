import codecs
import json
import openai
import os
import pypandoc

class GPTGrader():

    def __init__(self, src_dir, dest_dir):
        """Initialize the GPTGrader class.

        Args:
            src_dir (string): The path to the folder containing the student submissions. 
            dest_dir (string): The path to the folder where the feedback files will be saved.
        """
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        src_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.join(src_dir, ".."), "config.json")) as json_file:
            self.json_config = json.load(json_file)
            openai.api_key = self.json_config['openai_api_key']


    def get_student_essay(self, filename):
        """Read the student essay from the file.

        Args:
            filename (string): The path to the file containing the student essay.

        Returns:
            string: The student essay. 
        """
        content = ""
        with codecs.open(filename, 'r', 'utf-8') as f:
            content = f.read()
        return content


    def grade(self, student_essay):
        """Grade the student essay using the OpenAI API.

        Args:
            student_essay (string): The student essay to be graded.

        Returns:
            string: The feedback for the student essay.
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system", 
                "content": self.json_config['backstory']
            },
            {
                "role": "user", 
                "content": "Assume the following facts: " + self.json_config['rubric'] + 
                            "\nHow would you grade the following?\n\n" + student_essay
            }]
        )
        return response['choices'][0]['message']['content']
    

    def convert_docx_to_txt(self, src_file):
        """Convert a docx file to a txt file.

        Args:
            src_file (string): The path to the docx file to be converted.

        Returns:
            string: The path to the converted txt file.
        """
        src_file_path = os.path.join(self.src_dir, src_file)
        dest_file_path = os.path.join(self.dest_dir, os.path.splitext(src_file)[0]+".txt")
        pypandoc.convert_file(src_file_path, 'plain', outputfile=dest_file_path)
        return dest_file_path
    
    
    def start(self):
        """Start the grading process.
        """
        for file in os.listdir(self.src_dir):
            txt_file = self.convert_docx_to_txt(file)
            feedback = self.grade(self.get_student_essay(txt_file))
            with open(os.path.join(self.dest_dir, os.path.splitext(txt_file)[0]+"_feedback.txt"), 'w') as f:
                f.write(feedback)
            os.remove(txt_file)
