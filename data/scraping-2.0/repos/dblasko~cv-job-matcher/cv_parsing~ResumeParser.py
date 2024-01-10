import pdftotext
import openai
import re
import json


class ResumeParser:
    def __init__(self, OPENAI_API_KEY):
        # set GPT-3 API key from the environment vairable
        openai.api_key = OPENAI_API_KEY
        # GPT-3 completion questions
        self.prompt_questions = """Summarize the text below into a JSON with exactly the following structure {basic_info: {location, portfolio_website_url, linkedin_url, github_main_page_url, university, education_level (BS, MS, or PhD), graduation_year, graduation_month, majors, GPA, languages (a list of languages), skills (a list of skills)}, project_experience:[{project_name, project_discription}], work_experience: [{experience_level, job_title, company, location, duration, job_summary}]}
"""

    # Extract the content of a pdf file to string.

    def pdf2string(self: object, pdf) -> str:
        pdf = pdftotext.PDF(pdf)
        pdf_str = "\n\n".join(pdf)
        pdf_str = re.sub("\s[,.]", ",", pdf_str)
        pdf_str = re.sub("[\n]+", "\n", pdf_str)
        pdf_str = re.sub("[\s]+", " ", pdf_str)
        pdf_str = re.sub("http[s]?(://)?", "", pdf_str)

        return pdf_str

    # Base function for querying GPT-3. Send a request to GPT-3 with the passed-in function parameters and return the response object.

    def query_completion(
        self: object,
        prompt: str,
        engine: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        top_p: int = 1,
        frequency_penalty: int = 0,
        presence_penalty: int = 0,
    ) -> object:
        estimated_prompt_tokens = int(len(prompt.split()) * 1.6)
        estimated_answer_tokens = 2049 - estimated_prompt_tokens

        # FOR GPT3
        # response = openai.ChatCompletion.create(
        #     model=engine,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=temperature,
        #     max_tokens=min(4096 - estimated_prompt_tokens, max_tokens),
        #     top_p=top_p,
        #     frequency_penalty=frequency_penalty,
        #     presence_penalty=presence_penalty,
        # )

        # FOR CURIE
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=min(4096 - estimated_prompt_tokens, max_tokens),
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        return response

    # Query GPT-3 for the work experience and / or basic information from the resume at the PDF file path.

    def query_resume(self: object, pdf_str) -> dict:
        resume = {}
        # pdf_str = self.pdf2string(pdf)
        # print(pdf_str)
        prompt = self.prompt_questions + "\n" + pdf_str
        max_tokens = 4090 - 864
        engine = "text-davinci-002"  # "gpt-3.5-turbo"
        response = self.query_completion(prompt, engine=engine, max_tokens=max_tokens)

        response_text = response["choices"][0]["text"].strip()
        # response_text = response["choices"][0]["message"][
        #     "content"
        # ].strip()  # if we ant to use gpt 3.5-turbo
        # print(response_text)
        resume = json.loads(response_text)
        # print(resume)
        return resume
