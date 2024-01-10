import os
from docx import Document
from prompts.gen_activity_prompts import gen_act_chat_prompt, mod_gen_act_human_chat_prompt
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

gpt4 = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.5,
    streaming=True,
    model="gpt-3.5-turbo-16k"
)

palm_llm = ChatGooglePalm(
    google_api_key=google_api_key,
    streaming=True,
    temperature=0.4,
)

activity_memory = ConversationBufferWindowMemory(k=10)

act_gen_chain = LLMChain(
    llm=gpt4,
    prompt=gen_act_chat_prompt,
    memory=activity_memory,
    output_key="activity"
)

gen_act_input_variables = ["subject", "details"]

mod_act_gen_chain = LLMChain(
    llm=gpt4,
    prompt=mod_gen_act_human_chat_prompt,
    memory=activity_memory,
    output_key="activity"
)


class ClassActivity:
    """
    This class represents a Classroom activity, generated using LLMs and Langchain,
    the python library and it's wrappers.
    """
    act_gen_chain = act_gen_chain
    mod_act_gen_chain = mod_act_gen_chain

    def __init__(self):
        self.act_gen_chain = act_gen_chain
        self.mod_act_gen_chain = mod_act_gen_chain

    def generate_activity(self):
        try:
            subject = input("Enter the subject: ")
            details = input("Enter the details: ")

            activity = self.act_gen_chain.generate([{
                "subject": subject,
                "details": details
            }])
            activity.flatten()

            activity_output = activity.generations[0][0].text
            return activity_output
        except Exception as e:
            print(f"Error generating activity: {e}")

    def modify_activity(self, activity: str):
        try:

            new_details = input("Enter new details: ")
            mod_gen_activity = self.mod_act_gen_chain.generate([{
                "activity": activity,
                "new_details": new_details
            }])
            modded_activity = mod_gen_activity.generations[0][0].text
            return modded_activity
        except Exception as e:
            print(f"Error modifying activity: {e}")

    @staticmethod
    def save_activity(activity, file_path):
        try:
            doc = Document()
            doc.add_paragraph(activity)
            doc.save(file_path)
            print(f"Activity saved to {file_path}")
        except Exception as e:
            print(f"Error saving activity: {e}")


test = ClassActivity()
test_output = test.generate_activity()
print(test_output)


modified_activity = test.modify_activity(test_output)
print(modified_activity)

# Enter the subject: Science
"""Please generate an activity to help teach my 4th grade students about the water cycle, 
preferably without making too much of a mess. They will be working in small groups of 4-5 students."""
