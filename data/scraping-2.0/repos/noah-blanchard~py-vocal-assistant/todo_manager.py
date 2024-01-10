from openai_agent import OpenAIAgent
from speech_processing import SpeechProcessing
from command_processing import CommandProcessing

class TodoManager:
    def __init__(self):
        self.openai_agent = OpenAIAgent()
        self.tasks = ["Buy milk", "Buy chocolate", "Go play football"]
        self.speech_processor = SpeechProcessing()
        self.command_processor = CommandProcessing()
    
    def handle_command(self, command):
        label = self.openai_agent.get_todo_command_label(command)
        print(f"Label: {label}, Command: {command}")
        if label == "add":
            self.add_to_todo_list(command)
        elif label == "list":
            self.get_todo_list()
        elif label == "remove":
            self.remove_from_todo_list(command)
        else:
            self.speech_processor.speak("I couldn't understand your command! Please try again.")

    def add_to_todo_list(self, item):
        
        todo = self.openai_agent.generated_todo(item)
        print(f"Todo to be added: {todo}")

        if todo:
            self.tasks.append(todo)
            self.speech_processor.speak(f"Succesfully added '{todo}' to your todo list !")
    
    def get_todo_list(self):
        self.speech_processor.queue("Here's what's in your todo list!")

        for index, task in enumerate(self.tasks):
            self.speech_processor.queue(f"{index + 1}: {task}", False)
        
        self.speech_processor.runAndWait()

    def remove_from_todo_list(self, command):
        task = self.openai_agent.recognize_todo(self.tasks, command)
        print(command, task)

        if task not in self.tasks:
            self.speech_processor.speak("I couldn't find the specified task in your to-do list. Please try again.")
        else:
            self.speech_processor.speak(f"Do you want to remove '{task}' from your to-do list ?")
            decision = self.speech_processor.listen()
            decision = self.command_processor.get_approve_deny(decision)

            print(decision)

            if decision == "approve":
                self.tasks.remove(task)
                self.speech_processor.speak(f"I have removed '{task}' from your to-do list.")
            else:
                self.speech_processor.speak(f"Okay! I won't remove '{task}' from your to-do list.")
