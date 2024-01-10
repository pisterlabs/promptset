from typing import Tuple
import openai
from statemachine import StateMachine, State # type: ignore
from entities import Prompt, Task
from gpt_classifier import get_response_and_append_it

with open("prompts/system.txt", "r") as f:
    system_prompt = f.read()

class Classifier(StateMachine):
    # states
    initializing = State("initializing", initial=True)
    waiting = State("waiting")
    exiting = State("exited")
    classifying_item = State("classify_item")
    classifying_list = State("classify_list")

    # commands
    
    
    set_task = initializing.to(waiting)
    print_state_initializing = initializing.to(initializing)
    just_chat_initializing = initializing.to(initializing)
    
    classify_item = waiting.to(classifying_item)
    list_classes = waiting.to(waiting)
    finish = waiting.to(exiting)
    print_state_waiting = waiting.to(waiting)
    just_chat_waiting = waiting.to(waiting)
    
    refine_item = classifying_item.to(classifying_item)
    accept_item_class = classifying_item.to(classifying_list)
    print_state_classifying_item = classifying_item.to(classifying_item)
    just_chat_classifying_item = classifying_item.to(classifying_item)
    
    refine_list = classifying_list.to(classifying_list)
    accept_list_class = classifying_list.to(waiting)  
    print_state_classifying_list = classifying_list.to(classifying_list)
    just_chat_classifying_list = classifying_list.to(classifying_list)
    
    print_state_exiting = exiting.to(exiting)
    just_chat_exiting = exiting.to(exiting)
    
    def print_state(self):
        if self.current_state == self.initializing:
            self.print_state_initializing()
        elif self.current_state == self.waiting:
            self.print_state_waiting()
        elif self.current_state == self.classifying_item:
            self.print_state_classifying_item()
        elif self.current_state == self.classifying_list:
            self.print_state_classifying_list()
        elif self.current_state == self.exiting:
            self.print_state_exiting()
    
    def just_chat(self):
        if self.current_state == self.initializing:
            self.just_chat_initializing()
        elif self.current_state == self.waiting:
            self.just_chat_waiting()
        elif self.current_state == self.classifying_item:
            self.just_chat_classifying_item()
        elif self.current_state == self.classifying_list:
            self.just_chat_classifying_list()
        elif self.current_state == self.exiting:
            self.just_chat_exiting()
            
    

start_prompt: Prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "assistant", "content": "Hello, I'm Gotcha."},
]


def set_task(starting_prompt: Prompt, task: Task) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    task_message = task.render()
    prompt.append({"role": "user", "content": task_message})
    return get_response_and_append_it(prompt)


def print_state(starting_prompt: Prompt) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": "PRINT STATE"})
    return get_response_and_append_it(prompt)

def just_chat(starting_prompt: Prompt, user_message: str) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": user_message})
    return get_response_and_append_it(prompt)
    
def classify_item(starting_prompt: Prompt) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": "CLASSIFY ITEM"})
    return get_response_and_append_it(prompt)
    
def list_classes(starting_prompt: Prompt) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": "LIST CLASSES"})
    return get_response_and_append_it(prompt)

def finish(starting_prompt: Prompt) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": "FINISH"})
    return get_response_and_append_it(prompt)

def refine_item(starting_prompt: Prompt, new_class: str, justification: str) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": f"REFINE ITEM(CLASS={new_class}, JUSTIFICATION={justification})"})
    return get_response_and_append_it(prompt)

def accept_item_class(starting_prompt: Prompt) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": "ACCEPT ITEM"})
    return get_response_and_append_it(prompt)

def refine_list(starting_prompt: Prompt, item_to_remove: str, justification: str) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": f"REFINE LIST(REMOVE={item_to_remove}, JUSTIFICATION={justification})"})
    return get_response_and_append_it(prompt)    
    
    
def accept_list_class(starting_prompt: Prompt) -> Tuple[str, Prompt]:
    prompt = starting_prompt.copy()
    prompt.append({'role': 'user', "content": "ACCEPT LIST"})
    return get_response_and_append_it(prompt)



def run_machine_step(prompt: Prompt, classifier: Classifier) -> Prompt:
    command = input("$>")
    if command == "set task":
        description = input("Enter description $>")
        task = Task(description, [], [])
        print("Enter examples.")
        task.read_examples()
        print("Enter unclassified items.")
        task.read_unclassified()
        message, out_prompt = set_task(prompt, task)
        print(f"Gotcha says$> {message}")
        classifier.set_task()
    elif command == "read task":
        filename = input("Enter filename $>")
        task = Task.from_file(filename)
        message, out_prompt = set_task(prompt, task)
        classifier.set_task()
    elif command == "print state":
        message, out_prompt = print_state(prompt)
        print(f"Gotcha says$> {message}")
        classifier.print_state()
    elif command == "classify item":
        message, out_prompt = classify_item(prompt)
        print(f"Gotcha says$> {message}")
        classifier.classify_item()
    elif command == "list classes":
        message, out_prompt = list_classes(prompt)
        print(f"Gotcha says$> {message}")
        classifier.list_classes()
    elif command == "accept item":
        message, out_prompt = accept_item_class(prompt)
        print(f"Gotcha says$> {message}")
        classifier.accept_item_class()
    elif command == "accept list":
        message, out_prompt = accept_list_class(prompt)
        print(f"Gotcha says$> {message}")
        classifier.accept_list_class()
    elif command == "refine item":
        new_class = input("Enter new class $>")
        justification = input("Enter justification $>")
        message, out_prompt = refine_item(prompt, new_class, justification)
        print(f"Gotcha says$> {message}")
        classifier.refine_item()
    elif command == "refine list":
        item_to_remove = input("Enter item to remove $>")
        justification = input("Enter justification $>")
        message, out_prompt = refine_list(prompt, item_to_remove, justification)
        print(f"Gotcha says$> {message}")
        classifier.refine_list()
    elif command == "finish":
        message, out_prompt = finish(prompt)
        print(f"Gotcha says$> {message}")
        classifier.finish()
    elif command == "quit":
        quit()    
    else:
        message, out_prompt = just_chat(prompt, command)
        print(f"Gotcha says$> {message}")
        classifier.just_chat()
        
    return out_prompt
    
        
if __name__ == "__main__":
    prompt = start_prompt.copy()
    classifier = Classifier()
    while True:
        prompt = run_machine_step(prompt, classifier)