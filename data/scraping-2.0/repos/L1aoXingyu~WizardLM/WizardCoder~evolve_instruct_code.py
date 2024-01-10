import random
from openai_gpt_request import send_prompt

class AddConstraints:
    def __init__(self):
        with open('code_prompts/add_constraints.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class Concretizing:
    def __init__(self):
        with open('code_prompts/concretizing.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class IncreaseReasoningSteps:
    def __init__(self):
        with open('code_prompts/increasing_reasoning_steps.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class Misleading:
    def __init__(self):
        with open('code_prompts/misleading.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class TimeSpaceComplexity:
    def __init__(self):
        with open('code_prompts/time_space_complexity.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class RandomSystemPrompt:
    def __init__(self) -> None:
        self.prompts = []
        with open('code_prompts/add_constraints.txt', 'r') as f:
            prompt = f.read()
            prompt = prompt.replace(r'{instruction}', '')
            self.prompts.append(prompt)
        with open('code_prompts/concretizing.txt', 'r') as f:
            prompt = f.read()
            prompt = prompt.replace(r'{instruction}', '')
            self.prompts.append(prompt)
        with open('code_prompts/increasing_reasoning_steps.txt', 'r') as f:
            prompt = f.read()
            prompt = prompt.replace(r'{instruction}', '')
            self.prompts.append(prompt)                    
        with open('code_prompts/misleading.txt', 'r') as f:
            prompt = f.read()
            prompt = prompt.replace(r'{instruction}', '')
            self.prompts.append(prompt)    
        with open('code_prompts/time_space_complexity.txt', 'r') as f:
            prompt = f.read()
            prompt = prompt.replace(r'{instruction}', '')
            self.prompts.append(prompt)

    def __call__(self) -> str:
        return random.choice(self.prompts)


if __name__ == '__main__':
    add_constraints = AddConstraints()
    concretizing = Concretizing()
    increase_reasoning_steps = IncreaseReasoningSteps()
    misleading = Misleading()
    time_space_complexity = TimeSpaceComplexity()

    print('-' * 80)
    print(add_constraints('Write a function that takes a list of numbers and returns the sum of the list.'))
    print('-' * 80)
    print(concretizing('Write a function that takes a list of numbers and returns the sum of the list.'))
    print('-' * 80)
    print(increase_reasoning_steps('Write a function that takes a list of numbers and returns the sum of the list.'))
    print('-' * 80)
    print(misleading('Write a function that takes a list of numbers and returns the sum of the list.'))
    print('-' * 80)
    print(time_space_complexity('Write a function that takes a list of numbers and returns the sum of the list.'))
