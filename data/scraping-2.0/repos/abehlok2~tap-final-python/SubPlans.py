import os

import docx
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

openai_api_key = os.environ['OPENAI_API_KEY']

gpt4 = ChatOpenAI(
  openai_api_key  = openai_api_key,
  temperature = 0.1,
  max_tokens = 100,
  model = "gpt4"
)

gpt3 = ChatOpenAI(
  openai_api_key = openai_api_key,
  temperature = 0.1,
  model = "gpt-3.5-turbo"
)

subplan_template_variables = ["Reading_topic", "Writing_topic", "Math_topic", \
                              "Science_topic", "Social_Studies_topic", \
                              "Special_Subject",
                              "additional_information"]                 

subplan_prompt_template = PromptTemplate(
  template = """As an AI language model, your task is to create a detailed set of \
  substitute teacher plans for a 4th grade class in Monroe County, New York State. The\
  plans should be of high quality, align with the curriculum standards for this grade \
  and location, and be tailored to the needs of the full-time teacher.The lesson \
  plans should cover the following subjects and topics: Reading: {Reading_topic} \
\
Writing: {Writing_topic}Math: {Math_topic} Science: {Science_topic} Social Studies: \
{Social_Studies_topic} Also, include guidance for handling the special subject of the\
day: {Special_Subject}. The sub will not be teaching these subjects, so simply mention\
that the class has them and relate them to the daily class schedule. The full-time \
teacher has also provided the following additional information about the class or \
specific students: {additional_information}. Ensure the plans are comprehensive, \
easy to follow, and provide all necessary information for a substitute teacher to \
effectively teach the class.""",
  input_variables = subplan_template_variables,
)

from fuzzywuzzy import process


class Subplans:
    """
    Class representing a set of subplans, generated for the teacher's sub.
    """
    SPECIALS = ["art", "gym", "music"]

    def __init__(self):
        self.subjects = ["reading", "writing", "math", "science", "social studies"]
        self.specials = {}
    def get_topics_from_user(self):
        self.topics = {}
        for x in self.subjects:
            x=-1
            subject_name = self.subjects[x++1]
            if x<=len(self.subjects):
                continue
            else:
                break
    
            topic = input(f"Please provide a topic for {subject_name}: ").lower().strip()
            self.topics[subject_name] = topic

        return self.topics

    def get_additional_information(self):
        additional_information = input("Please provide any additional information about\
        the class or specific students: ")
        return additional_information

    def get_specials(self):
        specials_input = input("Please provide any specials for the day: ").lower().split(',')
        specials_input = [special.strip() for special in specials_input]

        for special in specials_input:
            match, score = process.extractOne(special, self.SPECIALS)
            if score > 80:
                self.specials[match] = special
            else:
                print(f"Sorry, I couldn't find a match for '{special}'. Please try again.")

        return self.specials

    def generate_subplans(self): 
        subplan_llmchain = LLMChain(
            llm = gpt3, 
            prompt = subplan_prompt_template,
        )

        subplan_prompt_output_data = subplan_llmchain.generate(
            input_list =  [{
                "Reading_topic": self.subjects[0],
                "Writing_topic": self.subjects[1],
                "Math_topic": self.subjects[2],
                "Science_topic": self.subjects[3],
                "Social_Studies_topic": self.subjects[4],
                "Special_Subject": ', '.join(self.specials.keys()),
                "additional_information": self.get_additional_information()
            }]
        ) 

        for generation in subplan_prompt_output_data.generations:
            chat_output = generation[0].text
            return chat_output

    def generate_complete_subplans(self):
        subplans = Subplans()
        subplans.get_topics_from_user()
        subplans.get_specials()
        generated_subplans = subplans.generate_subplans()
        return generated_subplans
    
    def save_subplans(self, path:str, generated_subplans):
        subplan_doc = docx.Document(generated_subplans)
        if not os.path.exists(path):
            print("Save location does not exist!")
            print("Would you like to create a new folder? (y/n)")
            if input("> ").lower() == "y":
                os.mkdir(path)
                print("Folder created!")
            else:
                print("Exiting...")
                return
        subplan_doc.save(path)