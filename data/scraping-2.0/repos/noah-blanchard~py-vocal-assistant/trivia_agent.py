import requests
import random
from speech_processing import SpeechProcessing
from command_processing import CommandProcessing
from openai_agent import OpenAIAgent

class TriviaAgent:
    def __init__(self):
        self.base_url = "https://the-trivia-api.com/v2/questions"
        self.speech_processor = SpeechProcessing()
        self.command_processor = CommandProcessing()
        self.openai_agent = OpenAIAgent()

    def handle_command(self, command):
        self.speech_processor.speak("Do you want me to ask you a trivia question ?")
        decision = self.speech_processor.listen()
        decision = self.command_processor.get_approve_deny(decision)

        if decision == "approve":
            self.speech_processor.speak("Ok! Let's play trivia !")
            self.start_trivia()
        else:
            self.speech_processor.speak("Ok, it is totally fine, let me know if you need help for anything.")
    
    def start_trivia(self):
        question = self.get_question()
        possible_answers = []
        possible_answers.append(question["correct"])
        possible_answers.extend(question["incorrect"])

        random.shuffle(possible_answers)

        self.speech_processor.queue(f"The category of the question is : {question['category']}")
        self.speech_processor.queue(f"Here's the question : {question['question']}", rephrase=False)

        for index, answer in enumerate(possible_answers):
            self.speech_processor.queue(f"{index + 1}: {answer}", rephrase=False)
        
        self.speech_processor.queue("What is your answer ?")
        self.speech_processor.runAndWait()

        self.get_and_check_answer(question["correct"])
    
    def get_and_check_answer(self, correct_answer):

        print("correct answer : ", correct_answer)

        answer = self.speech_processor.listen()
        verdict = self.openai_agent.check_trivia_answer(correct_answer, answer)

        print("verdict : ", verdict)

        if verdict.lower() == "correct":
            self.speech_processor.speak("Congratulations! This is the right answer to the trivia question.")
        else:
            self.speech_processor.speak("This isn't the right answer to the trivia question. You can try a new one if you want.")


    def get_question(self, limit=1):
        try:
            params = {
                "limit": limit
            }

            response = requests.get(self.base_url, params=params)

            if response.status_code == 200:
                data = response.json()[0]
                question_data = {
                    "category": data["category"],
                    "question": data["question"]["text"],
                    "correct": data["correctAnswer"],
                    "incorrect": data["incorrectAnswers"]
                }
                return question_data

        except Exception as e:
            print("There was an error retrieving a question : ", e)
        return None