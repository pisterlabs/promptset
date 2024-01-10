import PyPDF2
import openai
import re
import logging
import json
from server.utils.tokenizer import num_tokens_from_string
from flask import Flask, request, jsonify
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app=Flask(__name__)

class ResumeParser():
    def __init__(self, OPENAI_API_KEY):
        # set GPT-3 API key from the environment vairable
        openai.api_key = OPENAI_API_KEY
        # GPT-3 completion questions
        self.prompt_questions = \
"""Summarize the text below into a JSON with exactly the following structure {basic_info: {first_name, last_name, full_name, email, phone_number, location, portfolio_website_url, linkedin_url, github_main_page_url, university, education_level (BS, MS, or PhD), graduation_year, graduation_month, majors, GPA}, work_experience: [{job_title, company, location, duration, job_summary}], project_experience:[{project_name, project_description}]}
"""
       # set up this parser's logger
        logging.basicConfig(filename='logs/parser.log', level=logging.DEBUG)
        self.logger = logging.getLogger()

    def pdf2string(self, pdf_path):
        """
        Extract the content of a pdf file to string.
        """
        # Open the PDF file using PyPDF2
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Read text from all pages
            pdf_str = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
            
            # Modify text formatting
            pdf_str = re.sub('\s[,.]', ',', pdf_str)
            pdf_str = re.sub('[\n]+', '\n', pdf_str)
            pdf_str = re.sub('[\s]+', ' ', pdf_str)
            pdf_str = re.sub('http[s]?(://)?', '', pdf_str)
            
            return pdf_str

    def query_completion(self: object,
                        prompt: str,
                        engine: str = 'text-curie-001',
                        temperature: float = 0.0,
                        max_tokens: int = 100,
                        top_p: int = 1,
                        frequency_penalty: int = 0,
                        presence_penalty: int = 0) -> object:
        """
        Base function for querying GPT-3. 
        Send a request to GPT-3 with the passed-in function parameters and return the response object.
        :param prompt: GPT-3 completion prompt.
        :param engine: The engine, or model, to generate completion.
        :param temperature: Controls the randomnesss. Lower means more deterministic.
        :param max_tokens: Maximum number of tokens to be used for prompt and completion combined.
        :param top_p: Controls diversity via nucleus sampling.
        :param frequency_penalty: How much to penalize new tokens based on their existence in text so far.
        :param presence_penalty: How much to penalize new tokens based on whether they appear in text so far.
        :return: GPT-3 response object
        """
        self.logger.info(f'query_completion: using {engine}')

        estimated_prompt_tokens = num_tokens_from_string(prompt, engine)
        estimated_answer_tokens = (max_tokens - estimated_prompt_tokens)
        self.logger.info(f'Tokens: {estimated_prompt_tokens} + {estimated_answer_tokens} = {max_tokens}')

        response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=estimated_answer_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
        )
        return response
    
    def query_resume(self: object, pdf_path: str) -> dict:
        """
        Query GPT-3 for the work experience and / or basic information from the resume at the PDF file path.
        :param pdf_path: Path to the PDF file.
        :return dictionary of resume with keys (basic_info, work_experience).
        """
        resume = {}
        pdf_str = self.pdf2string(pdf_path)
        pdf_str = pdf_str.encode('ascii', 'ignore').decode('ascii')
        prompt = self.prompt_questions + '\n' + pdf_str
        engine = 'text-davinci-002'
        max_tokens = 4097
        response = self.query_completion(prompt, engine=engine, max_tokens=max_tokens)
        response_text = response.choices[0]['text'].strip()
        print(response_text)
        return response_text

    
@app.route('/parse-resume', methods=['POST'])
def parse_resume():
    try:
        parser = ResumeParser('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')
        pdf_file = request.files['resume']
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            pdf_path = os.path.join('uploads', pdf_file.filename)
            pdf_file.save(pdf_path)
            resume_data = parser.query_resume(pdf_path)  # Retrieve the parsed resume data
            return jsonify({"resume": resume_data})  # Return the parsed resume data in JSON format
        else:
            return jsonify({"error": "Invalid file format. Only PDF files are supported."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(port=3000, debug=True)