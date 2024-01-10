from openai import OpenAI

class JobDescriptionWriter:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def read_cv(self):
        with open('cv.txt','r')as file:
            cv= file.read()
            return cv

    def compose_presentation_letter(self, description):

        cv = self.read_cv
        print(cv)
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an eloquent writer with skills to write excellent and persuasive texts. (MAXIMUM 50 WORDS.)"},
                {"role": "user", "content": f"You are a human resource expert responsible for creating presentation letters tailored to specific job descriptions. Your task is to generate a presentation letter based on the skills listed on the candidate's CV, without fabricating any additional skills. Here is the CV: {cv} \nCraft a compelling letter that highlights the candidate's relevant experience, achievements, and qualities, ensuring it aligns with the specific requirements of the job description: {description}. \nAnswer with a maximum of 50 words."}
            ]
        )
        return completion.choices[0].message.content
    
    def evaluation_cv(self, cv, description):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an Human ressources Manager that analyze CVs for jobs requests.(MAXIMUM 50 WORDS.)"},
                {"role": "user", "content": f"You are a highly experienced human resource manager responsible for evaluating and analyzing CVs for various job positions. Today, you have been assigned the task of reviewing a CV for a [specific job request]. Your objective is to provide a comprehensive note on a scale of 1 to 10, highlighting three strengths and three weaknesses of the candidate, along with an overall analysis. Remember to consider the candidate's suitability for the [specific job request] position and provide constructive feedback to help improve their chances of success. Here is the cv: {cv}, \n and here is the description job: {description}, \nAnswer with maximum of 20 Words."}
            ]
        )
        return completion.choices[0].message.content
