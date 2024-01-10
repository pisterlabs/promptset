
import os
import openai
from dotenv import dotenv_values
import logging
from pathlib import Path

class GPT3Connector():
    """ Text completion using the OpenAI GPT-3 API (see https://beta.openai.com/docs/api-reference/introduction) """

    def __init__(self) -> None:
        curr_path = Path(__file__).resolve().parent
        filepath = curr_path.joinpath(".env")
        config = dotenv_values(filepath)
        if "OPENAI_API_KEY" in config:
            openai.api_key = config["OPENAI_API_KEY"]
        else:
            logging.warning("Couldn't find OPENAI API Key.")
            raise RuntimeError() 


    def _limit_input_length(self, text) -> str:
        """ Limit input length to limit the amount of used tokens. (see https://beta.openai.com/docs/usage-guidelines/safety-best-practices) """
        # Hint: should actually be already done in front-end
        required_input_tokens = len(text) / 4

        if required_input_tokens >= 100:
            text = text[:395]
        return text

    def _generate_prompt_example(self, animal):
        return """Suggest three names for an animal that is a superhero.

        Animal: Cat
        Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
        Animal: Dog
        Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
        Animal: {}
        Names:""".format(
                animal.capitalize()
            )

    def _generate_prompt_summarize_tasks(self, text:str):
        return """Summarize the tasks given in the following text:\n
        '''\n
        {}\n
        '''\n
        The tasks are:\n
        -""".format(
                text
        )

    def _generate_prompt_guess_job_title(self, text:str):
        return """Guess my job title based on the following tasks:\n
        '''\n
        {}\n
        '''\n
        My job title is:""".format(
                text
        )

    def _generate_prompt_comment_job_title(self, text:str):
        return """The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n
        '''\n
        Human: What do you think is cool about the job {}?\n
        AI: I think\n
        '''
        """.format(
                text
        )

    def _generate_prompt_comment_technology_skill(self, text:str):
        return """The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n
        '''\n
        Human: I'm interested in {}.\n
        AI: {} is a nice skill to have because\n
        '''
        """.format(
                text, text
        )
    
    def gpt3_comment_technology_skill(self, text:str, more_than_one_skill:bool) -> str:
        """ Let GPT-3 comment on one of the extracted technology skills the user is interested in. """
        text = self._limit_input_length(text)
        response = openai.Completion.create(
            engine= "text-davinci-002",   #"text-davinci-002",
            prompt=self._generate_prompt_comment_technology_skill(text),
            temperature=0.6,
            max_tokens=256,
            top_p=1,
            n=1
        )
        comment:str = response.choices[0].text
        comment = comment.replace('\n', '').strip()

        # Tune output
        if not comment.startswith(text):
            if not comment[0:1]=="I ":
                comment = comment[0].lower() + comment[1:]
            if comment.startswith("it is"):
                comment = text + " is cool because " + comment
        if more_than_one_skill:
            comment = "especially " + comment + " 🤓"
        else:
            comment = "I think " + comment + " 🤓"
        return comment

    def gpt3_summarize_tasks(self, text:str) -> str:
        """ Let GPT-3 extract work tasks from the input text and list it in bullet points. """
        text = self._limit_input_length(text)
        response = openai.Completion.create(
            engine= "text-curie-001",   #"text-davinci-002",
            prompt=self._generate_prompt_summarize_tasks(text),
            temperature=0.6,
            max_tokens=256,
            top_p=1,
            n=1
        )

        tasks = str(response.choices[0].text)
        tasks = "-" + tasks
        tasks = tasks.strip()
        return tasks

    def gpt3_guess_job_title(self, text:str) -> str:
        """ Let GPT-3 guess the job title corresponding to tasks given as input. """
        text = self._limit_input_length(text)
        response = openai.Completion.create(
            engine= "text-davinci-002",
            prompt=self._generate_prompt_guess_job_title(text),
            temperature=0.6,
            max_tokens=256,
            top_p=1,
            n=1
        )

        job_title_guess:str = response.choices[0].text
        job_title_guess = (job_title_guess.replace('.', '').replace('\n', '').title()).strip()
        return job_title_guess

    def gpt3_comment_job_title(self, text:str) -> str:
        """ Let GPT-3 guess the job title corresponding to tasks given as input. """
        text = self._limit_input_length(text)
        response = openai.Completion.create(
            engine= "text-davinci-002",   #"text-davinci-002",
            prompt=self._generate_prompt_comment_job_title(text),
            temperature=0.75,
            max_tokens=200,
            top_p=1,
            n=1
        )
        
        # postprocess GPT-3 output
        comment:str = response.choices[0].text
        comment = comment.replace('\n', '').strip()
        if comment.startswith("Ai:"):
            comment = comment[3:]
        if comment.startswith("aI: "):
            comment = comment[4:]
        if not comment[0]=="I":
            comment = comment[0].lower() + comment[1:]
        if comment.startswith("that"):
            comment = comment + "it's cool "
        comment = "In my opinion, " + comment
        comment = comment + " 🦾🤖"

        return comment

    

# Test individual methods


# #%%
# connector = GPT3Connector()

# #%%
# response = connector.gpt3_summarize_tasks("Communicate and coordinate with management, shareholders, customers, and employees to address sustainability issues. Enact or oversee a corporate sustainability strategy.")
# #%%
# job_title_guess = connector.gpt3_guess_job_title(response)
# #%%
# comment = connector.gpt3_comment_job_title("Software Engineer")
# # %%
# comment = connector.gpt3_comment_job_title("Corporate Sustainability Officer")
# print(comment)
# #%%
# comment = connector.gpt3_comment_job_title(job_title_guess)
# print(comment)

# #%%
# comment = connector.gpt3_comment_technology_skill("Excel", more_than_one_skill=False)
# print(comment)
