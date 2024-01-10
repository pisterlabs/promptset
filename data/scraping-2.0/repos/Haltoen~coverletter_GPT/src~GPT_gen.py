import os
import json
import openai
import database
import deepTranslator
db = database.db()

class GPT_Handler():
    """
    This class handles prompt building and GPT API calls
    """
    def __init__(self):
        self.has_api_key = True
        self.app = None
        self.trans = deepTranslator.DeepTrans()

    def ref_app(self, app):
        self.app = app

    def add_API_KEY(self, api_key):
        """
        Add API key as environment variable.
        """
        match api_key:
            case type(str):
                 os.environ['API_KEY'] = api_key
            case type(tuple):
                 os.environ['API_KEY'] = api_key[0]

        self.activate_API_KEY()

    def activate_API_KEY(self):
        """
        Activates api key from environment
        """
        api_key = os.environ.get('API_KEY')
        try:
            openai.api_key = api_key
            self.has_api_key = True
            print("API key connected.")
        except ValueError:
            self.has_api_key = False
            print("Failed to connect to API with the provided API key.")


    def build_task_prompt(self, task: str, context: dict) -> str:
        prompt = f"Task: {task}\n\nContext:"
        for key in context.keys():
            prompt = prompt + f"{key}: {context[key]}\n"
        return prompt

    def chat_complete(self, prompt: str, temperature=0.7, max_tokens=300, funcs=None):
        msg =[]
        msg.append({"role": "user",
        "content": prompt})

        print("attempting to prompt model with:", msg)
        if funcs:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = msg,
                temperature = temperature,
                max_tokens = max_tokens,
                functions = funcs
            )
        else:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [{"role": "user",
                            "content": prompt}],
                temperature = temperature,
                max_tokens = max_tokens,
            )
        return response
    
    def make_coverletter(self, input : tuple):
        language = input[0]
        employer = input[1]
        job_description = input[2]
        extra_info = input[3] # (bool) let gpt ask for additional info
        skills = db.skill_list()
        resume = db.get_resume(language)
        dict_context = {
            "Employer": employer,
            "Job description": job_description,
            "skills": skills,
            "resume": resume
        }

        if extra_info:
            extra_prompt = self.build_task_prompt(task=
                """
                Spot vital skills in the job description that might not be apparent.
                Share input on crucial skills for a tailored cover letter, excluding those already evident.
                Analyze the description to identify key abilities, such as communication and problem-solving, that merit emphasis.
                """,
                context=dict_context
            )
            
            func ={
                "name": "skill_list",
                "description": "Create a list of relevant skills for your task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lst": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "relevant skills"
                        }
                    },
                    "required": ["lst"]
                }
            }
        
            outfunc = self.chat_complete(prompt=extra_prompt, funcs=[func])

            # Parse the JSON response
            response_data = json.loads(str(outfunc))

            # Extract function call information
            function_call_info = response_data["choices"][0]["message"]["function_call"]
            function_arguments = json.loads(function_call_info["arguments"])
            self.app.skill_popup(function_arguments["lst"]) # Run function
            dict_context["skills"] = db.skill_list()

        main_prompt = self.build_task_prompt(task=
            """
            Create a tailored cover letter showcasing your skills, experience, and alignment with the job description.
            Address key requirements and ensure a structured, engaging letter.
            """,
            context=dict_context
        )
                                            
        response = self.chat_complete(prompt=main_prompt, max_tokens=500)
        res_data = json.loads(str(response))
        content = res_data["choices"][0]["message"]["content"]
        content = self.trans.translate(dst=language, txt=content) #translation

        print(content)
        self.app.show_popup(title="Your Coverletter", text_field=content)


    

    






