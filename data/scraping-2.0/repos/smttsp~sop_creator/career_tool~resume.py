import json
import re

from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.prompts import ChatPromptTemplate

from career_tool.utils.file_utils import anonymize_resume, read_text_from_file
from career_tool.utils.word_utils import get_word_cloud, get_word_freq
from langchain.llms import VertexAI
from json import JSONDecodeError


class Resume:
    def __init__(self, resume_file):
        self.resume_file = resume_file
        self.content = self.get_resume_content()
        # self.json_resume = self.convert_to_json_resume()
        self.anonymized_resume = self.get_anonymized_resume()
        self.wc = get_word_cloud(self.content)
        self.wf = get_word_freq(self.wc)

    def get_resume_content(self):
        return read_text_from_file(self.resume_file)

    def get_anonymized_resume(self):
        return anonymize_resume(self.content)

    # def convert_to_json_resume(self):
    #     """Creating ChatGPT prompt to convert resume to JSON format."""
    #     prompt = (
    #         f"Given the resume here: {self.content}."
    #         "Can you convert it to JSON format?"
    #         "The JSON format should be as follows:"
    #         "{"
    #         "name: <name>\n"
    #         "email: <email>\n"
    #         "linkedin: <linkedin>\n"
    #         "github: <github>\n"
    #         "skills: <skills>\n"
    #         "experience: <experience>\n"
    #         "education: <education>\n"
    #         "projects: <projects>\n"
    #         "awards: <awards>\n"
    #         "publications: <publications>\n"
    #         "certifications: <certifications>\n"
    #         "languages: <languages>\n"
    #         "interests: <interests>\n"
    #         "references: <references>\n"
    #         "}"
    #         "If there are other fields, please add them as well."
    #         "Some of the fields may be empty."
    #         "Some of the fields such as experience, education may be list."
    #         "Don't add any text before or after the JSON format."
    #     )
    #     content = get_completion(prompt)
    #     resume_as_dict = json.load(content)
    #     return resume_as_dict

    # def convert_to_pdf_resume(self, template_name):
    #     pass


class ResumeAnalyzer:
    def __init__(self, resume, session_info, llm_model="gpt-3.5-turbo"):
        self.resume = resume
        self.session_info = session_info
        from time import time
        # t1 = time()
        self.recommendations = self._get_ai_recommendations_google()
        # print("vertex", time() - t1)

        # for model in []:
        # t1 = time()
        # self.recommendations = self._get_ai_recommendations(llm_model)

        # print("openai", time() - t1)
        # print()
        # self.recommendations = self._get_ai_recommendations(llm_model)
        # self.info = self._get_resume_details(llm_model)

        # self.name = info.get("name", "")
        # self.email = info.get("email", "")
        # self.phone_no = info.get("phone_no", "")
        # self.location = info.get("location", "")
        # self.github_address = info.get("github_address", "")
        # self.linkedin_address = info.get("linkedin_address", "")
        # self.fitting_job_title = info.get("fitting_job_title", "")
        # self.top5_skills = info.get("top5_skills", [])
        # self.total_industry_exp = info.get("total_industry_exp", 0)
        # self.total_academic_exp = info.get("total_academic_exp", 0)
        # self.management_score = info.get("management_score", 0)
        # self.professional_summary = info.get("professional_summary", "")

    def _get_ai_recommendations_google(self):
        template_string2 = """Given a resume ```{resume}```
        Create a list of changes you recommend to the resume.
        You need to look into the following things such as 

        1. the usage of action verbs
        2. addition of more quantifiable achievements. You can put them as "***" in your
            suggestions. (add comment  what user should put there)
        3. highlight the leadership experience. 
        4. the usage of numbers, ratios, improvements
        3. grammar, spelling, punctuation issues
        4. consistency of tense and other grammatical elements. Notice that current 
            experience should be present tense, past experience should be past tense.
        5. if you think some sections are redundant, such as interests, references, etc, 
            you can suggest to remove them.

        take a deep breath and give me suggestions in JSON format where the suggestions
        are in a list of dictionaries with the following keys:        
            - before:
            - after:
            - reason:
        """
        google_api_key = self.session_info.google_service_account

        try:
            chat = ChatVertexAI(
                # temperature=0.0,
                google_api_key=google_api_key,
                model="text-bison",
                max_output_tokens=2048,
            )

            prompt_template = ChatPromptTemplate.from_template(template_string2)

            service_messages = prompt_template.format_messages(
                resume=self.resume.content
            )
            response = chat(service_messages)
            json_match = re.search(r'```JSON(.*?)```', response.content, re.DOTALL)

            if json_match:
                json_data = json_match.group(1).strip()
                info = json.loads(json_data)
            else:
                info = []
                print("No JSON data found in the input string.")

        except JSONDecodeError | Exception as e:
            print(e)
            info = []
        return info

    def _get_ai_recommendations(self, llm_model="gpt-3.5-turbo"):
        template_string2 = """Given a resume ```{resume}```
        Create a list of changes you recommend to the resume.
        You need to look into the following things such as 
        
        1. the usage of action verbs
        2. addition of more quantifiable achievements. You can put them as "***" in your
            suggestions. (add comment  what user should put there)
        3. highlight the leadership experience. 
        4. the usage of numbers, ratios, improvements
        3. grammar, spelling, punctuation issues
        4. consistency of tense and other grammatical elements. Notice that current 
            experience should be present tense, past experience should be past tense.
        5. if you think some sections are redundant, such as interests, references, etc, 
            you can suggest to remove them.

        take a deep breath and give me suggestions in JSON format where the suggestions
        are in a list of dictionaries with the following keys:        
            - before:
            - after:
            - reason:
        """

        try:
            chat = ChatOpenAI(
                temperature=0.0,
                openai_api_key=self.session_info.openai_api_key,
                model=llm_model,
            )

            prompt_template = ChatPromptTemplate.from_template(template_string2)

            service_messages = prompt_template.format_messages(
                resume=self.resume.content
            )
            response = chat(service_messages)
            info = json.loads(response.content)
        except Exception as e:
            print(e)
            info = {}
        return info

    def _get_resume_details(self, llm_model="gpt-3.5-turbo"):
        chat = ChatOpenAI(temperature=0.0, model=llm_model)

        template_string = """Given a resume ```{resume}```
            Can you extract the followings information
                name
                email
                phone no (as xxx-xxx-xxxx if it is USA number, or similar format) 
                location
                github_address
                linkedin_address
                fitting_job_title
                top5_skills looking at the entire resume
                total_industry_experience (in years)
                total_academic_experience (in years)
                management_score (out of 100)
                professional_summary: Can you come up with a professional_summary which
                    is at most 100 tokens? This is a concise, memorable statement that 
                    lets the reader know what you offer the company
            ---
            Take a deep breath and give me the answer. 
            Format the output as JSON with the following keys:
                name
                email
                phone_no 
                location
                github_address
                linkedin_address
                fitting_job_title
                top5_skills
                total_industry_exp
                total_academic_exp
                management_score
                professional_summary
        """

        prompt_template = ChatPromptTemplate.from_template(template_string)

        service_messages = prompt_template.format_messages(
            resume=self.resume.content
        )
        response = chat(service_messages)
        recommendations = json.loads(response.content)

        return recommendations
