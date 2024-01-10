import numpy
import openai
from module import qr_model
import torch

class entity_tracker():
    def __init__(self, topic=None):
        self.history = []
        self.topic = topic
        self.qr_model = qr_model(device = torch.device("cuda:0"))
        
    def get_response(self, prompt):
        completion = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return completion.choices[0].text

    def topic_judge(self, user_utter):
        """
        decide whether the current topic of the user utterance is similar to the current topic
        """
        prompt = """Does the sentence contain the topic? Return True or False. 
            Example 1: 
                Sentence: What is DRAM (DRAM is Dynamic Random Access Memory, so it's a type of memory)
                Topic: Moore machine (Moore machine is a type of machine)
                Return: False (because the memory is different from a machine)
            Example 2:
                Sentence: Where is LA? (LA is Los Angeles, so it's a city)
                Topic: Place (Place is a type of location)
                Return: True (because the city is a type of location)
            Now your turn. 
            Sentence:
            """ + user_utter + "\nTopic: " + self.topic + "\nReturn:"
        text = self.get_response(prompt)
        if (text.find("True") != -1 or text.find("true") != -1) and self.topic != None:
            return True
        else:
            return False

    def determine_topic(self, user_utter):
        """
        determine the topic of the current user utterance
        """
        prompt = """
        Determine the topic of the sentence. 
        Example: 
        Sentence: What is Milly Machine? 
        Answer: Milly Machine
        Sentence: Who is Alan Turing? 
        Answer: Alan Turing
        Sentence: 
        """ + user_utter + "\nAnswer:"
        text = self.get_response(prompt)
        truncated = text.strip()
        return truncated
    
    def question_rewrite(self, user_utter):
        """
        re-write the question, like replacing the pronouns in the user utterance with the topic
        """
        pronoun_list = ["he", "she", "it", "they", "them", "He", "She", "It", "They", "Them"]
        pronouns_list = ["his", "her", "its", "their", "theirs", "His", "Her", "Its", "Their", "Theirs"]
        # if the user utterance contains a pronoun, replace it with the topic
        if any(pronoun in user_utter for pronoun in pronoun_list):
            for pronoun in pronoun_list:
                if pronoun in user_utter:
                    user_utter = user_utter.replace(pronoun, self.topic)
            for pronoun in pronouns_list:
                if pronoun in user_utter:
                    user_utter = user_utter.replace(pronoun, self.topic + "'s")
        return user_utter
    
    def answer_attach(self, answer):
        self.history.append(answer)
    
    def main(self, user_utter):
        # user_utter = self.question_rewrite(user_utter) ## modify the function before you activate this line
        user_utter = self.qr_model.qr(self.history,user_utter)
        if self.topic_judge(user_utter):
            self.history.append(user_utter)
        else:
            self.topic = self.determine_topic(user_utter)
            self.history = [user_utter]
        return user_utter, self.topic, self.history
        

def chatbot_answer(user_utter, context):
    return "This is a default answer."
        
        
if __name__ == "__main__":
    et = entity_tracker("turing machine")
    # ask the user to put in sentences
    while True:
        user_utter = input("Please enter a sentence: ")
        user_utter, topic, history = et.main(user_utter)
        answer = chatbot_answer(user_utter, context=history)
        et.answer_attach(answer)
        print("history =", history[:-1])
        print("topic =", topic)
        print("utter =", user_utter)

# q -> q(t),history[q(t-1),a],topic
