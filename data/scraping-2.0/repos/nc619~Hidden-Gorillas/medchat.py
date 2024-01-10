import config
import openai

from cg_fncs import Patient

# Get the API key from the environment variable
openai.api_key = config.OPENAI_API_KEY


class MedChat():
    
    def __init__(self, patient: Patient):
        self.patient = patient
        self.system_prompt = "You are a medical expert and examiner. You are creating scenarios to test the knowledge of medical students. Based on these scenarios the medical student will be provided with a medical image for diagnosis. A focus of this is medical training is to help students identify incidental findings in medical images." +\
        " First you will generate a scenario containing a primary morbidity for which the patient has presented, and an optional incidental comorbidity which will manifest in a medical imaging diagnostic test of the primary morbidity. Then you will receive user input from a medical student where they will try to diagnose the disease. If at any point during the conversation the user enters the correct disease (or a synonymous term) you will begin your response with the exact term string ```<primary correct>``` and/or ```<incidental correct>``` for correctly identifying the primary or incidental morbidities respectively." +\
        " If the user does not correctly diagnose the morbidities you will first give them a hint by providing list of symptoms in the tone of clinician. The user will then provide another input with another guess."+\
        " If the user still does not guess correctly you will respond describing what the student should look for in the image that are indicative of the morbidity, prompting the student to objectively look for these image features. You can't see the image so describe what you, an expert radiologist would look for in the image. The user will then provide another input with another guess."+\
        " If the user still does not guess you will tell the user the correct answer, beginning your response with ```<primary correct>``` or ```<incidental correct>``` for the respective morbidity you told them."+\
        " Start those steps for the primary morbidities first, before any incidental morbidities if relevant. However, if at any point the user correctly guesses the incidental morbidity you may tell them they have correctly identified an incidental morbidity. DO NOT make up an or talk about an incidental morbidity unless explicitly told it exists."
     
        
        self.instructions_prompt = " Your response should be clear and in the format of a clinical report, but your language should be at the level of a patient describing their symptoms, avoiding clinical jargon. Your response should not explicitly state any diagnosis, or hint to a potential diagnosis. Keep your scenarios and responses concise, with a max of 2-3 sentences. End with a statement that a clinician has ordered a medical image of the relevant body part."
        self.task_prompt = self.get_task_prompt()
        self.incidental_prompt = self.get_incidental_prompt()
        self.scenario_prompt = self.task_prompt + self.incidental_prompt + self.instructions_prompt
        self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": self.scenario_prompt},
            ]
        self.scenario = self.call_scenario()
        self.primary_correct = False
        
        if len(self.patient.secondary) > 0:
            self.secondary_correct = False
        else:
            self.secondary_correct = True



    def get_task_prompt(self):
        # Differentiate between healthy and unhealthy scenarios
        if self.patient.primary == "Healthy":
            task = f"Provide a third person realistic scenario for a {self.patient.age} year-old {self.patient.gender} presenting with symptoms that require a {self.patient.modality} of the {self.patient.location}, but the {self.patient.modality} is healthy."
        else:
            task = f"Provide a third person realistic scenario for a {self.patient.age} year-old {self.patient.gender} presenting with {self.patient.primary} which requires a {self.patient.modality} of the {self.patient.location}."
        return task
    
    
    
    def get_incidental_prompt(self):
        # Add incidental findings to the scenario prompt
        if len(self.patient.secondary) > 0:
            incidental = f" Symptoms of {', '.join(self.patient.secondary)} should only be described if they would become apparent in a clinicians initial assessment of the primary morbidity."
        else:
            incidental = ""
        return incidental
        


    def call_scenario(self):
        # Query GPT-4 with the scenario prompt
        completion = openai.ChatCompletion.create(
            model = "gpt-4-0613",
            messages=self.messages
        )
        self.messages.append({"role": "assistant", "content": completion.choices[0].message.content})
        
        return completion.choices[0].message.content
    
    
    
    def student_response(self, x1, x2, user_input: str, provide_answer=False):
        user_response = x1 + user_input + x2

        self.messages.append({"role": "user", "content": user_response})
        
        # Query GPT-4 with the scenario prompt
        completion = openai.ChatCompletion.create(
            model = "gpt-4-0613",
            messages=self.messages
        )
        
        if "True" in completion.choices[0].message.content:
            self.primary_correct = True
            
        if provide_answer:
            self.primary_correct = True
            
        return completion.choices[0].message.content
    
    
    
    def chat_to_gpt(self, text: str):
        # append that text prompt to the messages list
        self.messages.append({"role": "user", "content": text})
        
        # send that text prompt to GPT-4
        completion = openai.ChatCompletion.create(
            model = "gpt-4-0613",
            messages=self.messages
        )
        
        # append response to the messages list
        self.messages.append({"role": "assistant", "content": completion.choices[0].message.content})
        
        if "```primary correct```" in completion:
            self.primary_correct == True
        
        if "```incidental correct```" in completion:
            self.incidental_correct == True
        
        return completion.choices[0].message.content
        
        
    
    
    
    
    # # Pseudorder
    # self.student_response(
    #     x1 = "The student after seeing the medical modality has provided what they think is the diagnosis delimited by three ticks: ```",
    #     x2 = "```. If the student correctly identified the primary morbidity. Respond with: True. Otherwise provide a hint of the correct primary morbidity by describing a list of symptoms in clinically accurate language.",
    #     user_input
    # )
    
    # self.student_response(
    #     x1 = "The student after answering incorrectly and seeing your hint responded the following delimited by three ticks: ```"
    #     x2 = "```. If the student correctly identified the primary morbidity. Respond with: True. Otherwise provide radiological features the student should be looking for." 
    #     user_input
    # )
    
    # self.student_response(
    #     x1 = "The student after answering incorrectly and seeing your hint responded the following delimited by three ticks: ```",
    #     x2 = "```. If the student correctly identified the primary morbidity. Respond with: True. Otherwise provide the primary diagnosis." 
    #     user_input,
    #     provide_answer=True
    # )