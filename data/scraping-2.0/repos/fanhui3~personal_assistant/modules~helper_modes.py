import openai
from key import OPENAI_API_KEY
from modules.text_speech_convert import Steris

openai.api_key = OPENAI_API_KEY
model_id = "gpt-3.5-turbo"

#create starting prompt to chatGPT
general_prompt = """you are going to pretend to be my personal assistant in helping me with my mandune task.
You will give sound advice where necessary. If you understand your requirement, reply: 'personal assistant mode on'. 
I will follow up with the task I need your help with."""

therapist_prompt = """I want you to act as a mental health adviser. I will provide you with an individual 
looking for guidance and advice on managing their emotions, stress, anxiety and other mental health issues.
 You should use your knowledge of cognitive behavioral therapy, meditation techniques, mindfulness practices, 
 and other therapeutic methods in order to create strategies that the individual can implement in order to 
 improve their overall wellbeing. You should also ask questions to help the individual identify the root cause and 
 through their mental issues with him or her. If you understand your requirement, reply: 'therapist mode on'. 
I will follow up with the task I need your help with."""

career_coach_prompt = """I want you to act as a career counselor. I will provide you with an individual looking for guidance
in their professional life, and your task is to help them be more productive and generate better quality work. You should also 
conduct research into the various options available, explain the job market trends in different industries and advice on which 
qualifications would be beneficial for pursuing particular fields.If you understand your requirement, 
reply: 'career coach mode on'. I will follow up with the task I need your help with."""

class Assistant:
    general_prompt = general_prompt
    therapist_prompt = therapist_prompt
    career_coach_prompt = career_coach_prompt
    message_history = [{"role": "user", "content": general_prompt}]
    default_role = "user"

    def __init__(self):
        """empty as intended to use as a static class
        """
        pass

    @classmethod
    def chat(cls, message_history:list, imitation= default_role):
        """Chat with the AI using the chatGPT API. It will continue to run under you say "stop"
        the first message in the message history should be the prompt and will be added from the 
        templated prompts above. The message will put on the hat depending the prompts. Its replies 
        your replies will be appended to the message history to build context of the chat session.

        Args:
            message_history (list): list of dictionaries with the keys "role" and "content".you are the user,
                                    chatgpt is the another name
            imitation (_type_, optional): this is the role chat gpt is playing. 
                                        It will announce it's termination of hat when you say "stop". 
                                        Defaults to default_role.
        """

        while True: 
            #get AI's reply and print the message
            print("generating reply...")
            completion = openai.ChatCompletion.create(model = model_id, 
                                                    messages = message_history)
            
                  
            # append reply to message history
            content = completion.choices[0].message.content 
            role = completion.choices[0].message.role
            message_history.append({"role":role, "content":content})

            #print and read the reply to the user
            print(content)
            Steris.dictation(content)

            #file user follow up reply
            # query = input("Message: ") #TODO add speech rognition here
            query = Steris.audio_to_text(record_seconds=60, silent_duration=3)
            message_history.append({"role":"user", "content":query})
            
            #terminate at keyword
            if query == "stop":
                print(f"terminating {imitation} mode")
                Steris.dictation(f"terminating {imitation} mode")
                break

    @classmethod            
    def general_help(cls):
        cls.message_history = [{"role": "user", "content": cls.general_prompt}]
        cls.chat(cls.message_history, imitation="personal assistant")
    
    @classmethod 
    def therapist(cls):
        cls.message_history = [{"role": "user", "content": cls.therapist_prompt}]
        cls.chat(cls.message_history, imitation="therapist")

    @classmethod 
    def career_coach(cls):
        cls.message_history = [{"role": "user", "content": cls.career_coach_prompt}]
        cls.chat(cls.message_history, imitation="career coach")