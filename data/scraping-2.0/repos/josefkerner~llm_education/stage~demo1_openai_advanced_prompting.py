from langchain import PromptTemplate
from model.gpt.gpt_chat import GPT_turbo_model
from prompts.inContextExampleManager import inContextExampleManager

class OpenAIPrompter:
    def __init__(self):
        self.llm = GPT_turbo_model(
            cfg={
                'model_name': 'gpt-3.5-turbo-16k',
            }
        )

    def test_base_prompt(self):
        cv = """
        John Doe is a Data Scientist with 3 years of experience.
        He has worked on multiple projects in the past.
        He has a Master's degree in Computer Science.
        He is proficient in Python, SQL, and Machine Learning.
        He knows how to use TensorFlow,PyTorch and Computer Vision libs.
        """
        role = """
        As a Lead Analyst, you will be responsible for developing and implementing advanced data models and algorithms to enable our trading strategies. You will work closely with our trading, engineering, and quantitative research teams to identify opportunities for data-driven growth and optimization. You will be working with some of the brightest minds in the industry and have access to cutting-edge technology and resources.

 What would you work on?
•	Develop and implement advanced data models and algorithms to support high-frequency trading strategies.
•	Work with trading and quantitative research teams to identify opportunities for data-driven growth and optimization.
•	Analyze large and complex datasets to generate insights and identify trends.
•	Design dashboards and reports to track key metrics and performance indicators.
•	Collaborate with cross-functional teams to provide insights and recommendations that inform business decisions.
•	Stay up to date with the latest developments in data science and machine learning and apply them to our trading strategies.

What makes you a great candidate?
•	Advanced degree in Computer Science, Statistics, Mathematics, or a related field from a top-tier university.
•	Proven experience in a lead data science or analytics role in a trading or finance environment.
•	Experience with programming languages such as Python, R, or SQL.
•	Strong analytical and problem-solving skills.
•	Excellent communication and interpersonal skills.
•	Experience with machine learning, natural language processing, or computer vision.
•	Familiarity with high-frequency trading and financial markets is preferred.

        """
        cv_example = """
        Petr Novak is a Machine Learning Engineer with 3 years of experience.
        He has worked on multiple projects in the past.
        He has a Master's degree in Numerical methods in Physics and AI.
        He is proficient in Python, R, SQL, and Machine Learning.
        He knows how to use TensorFlow and PyTorch but mostly likes Keras.
        """
        prompt = f"""
        Generate a recommendation if candidate a fit for the job role based on role requirements and candidate experience.
        If a candidate is not a fit for the job role, generate only string : "Not a fit".
        Candidate name: John Doe,
        Candidate experience: {cv}
        Job role requirements: {role}
        Recommendation:
        """
        prompt_with_example = f"""
        Generate a recommendation if candidate a fit for the job role based on role requirements and candidate experience.
        If a candidate is not a fit for the job role, generate only string : "Not a fit".
        Example:
        ------------
        Candidate name: Petr Novak,
        Candidate experience: {cv_example}
        Job role requirements: {role}
        Recommendation: Petr is a fit for the job role based on his knowledge.
        ------------
        Candidate name: John Doe,
        Candidate experience: {cv}
        Job role requirements: {role}
        Recommendation:
        """
        prompt = prompt.replace('\n', '').replace('\t', '').replace('  ', '')
        prompt_with_example = prompt_with_example.replace('\n', '').replace('\t', '').replace('  ', '')
        message = [{"role": "user", "content": prompt}]
        #generate base prompt
        result = self.llm.generate([message])
        #generate prompt with single example
        message = [{"role": "user", "content": prompt_with_example}]
        result_with_example = self.llm.generate([message])
        print("Base prompt result:")
        print(result)
        print('-------------------------------')
        print("Prompt result with example:")
        print(result_with_example)

    def test_advanced_prompt(self):
        #cv
        #role
        #recomendation
        role = """
                We are looking for a Data Scientist with 3 years of experience.
                The candidate should have a Master's degree in Computer Science.
                The candidate should be proficient in Python, SQL, and Machine Learning.
                The candidate should know how to use TensorFlow and PyTorch.
                """
        cv_example = """
                Petr Novak is a Data Scientist with 3 years of experience.
                He has worked on multiple projects in the past.
                He has a Master's degree in Computer Science.
                He is proficient in Python, SQL, and Machine Learning.
                He knows how to use TensorFlow and PyTorch.
                """

        prompt = inContextExampleManager().get_prompt(
            role_summary=role,
            cv_summary=cv_example,
            name="Petr",
            surname="Novak",
        )
        print(prompt)
        prompt = prompt.replace('\n', '').replace('\t', '').replace('  ', '')
        message = [{"role": "user", "content": prompt}]
        result = self.llm.generate([message])
        print("Advanced prompt result:")
        print(result)


if __name__ == '__main__':
    prompter = OpenAIPrompter()
    #prompter.test_base_prompt()
    #uncomment to test advanced prompt
    prompter.test_advanced_prompt()

