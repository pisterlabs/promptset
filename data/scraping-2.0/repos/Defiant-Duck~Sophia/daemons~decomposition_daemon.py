from exploratory.openai_model import OpenAIModel

class DecompositionDaemon:
    def __init__(self, threshold=0.95,
                 prompt_template="You are an extremely competent and decisive classifier and resources are extremely scarce. That's why it's very important to resolve as many requests as possible without decomposing into tasks.  However, failure to decompose a complex request into tasks is also unacceptable. Precisely estimate the confidence (0-1) that the task '{task}' can be completed directly without decomposition. Please limit your response to a value that is valid as a Python float type. You must always return a float, and only a float.:"):
        self.model = OpenAIModel()
        self.threshold = threshold
        self.prompt_template = prompt_template

    def should_decompose(self, task):
        # Construct the prompt
        prompt = self.prompt_template.format(task=task)
        user_input = [{"role": "user", "content": prompt}]
        # Query the model
        confidence_estimate = self.model.generate_response(messages=user_input)
        print(confidence_estimate)

        # Determine if the task should be decomposed based on the confidence estimate
        return float(confidence_estimate['content']) < self.threshold

    def invoke(self, input):
        if self.should_decompose(input):
            return self.decompose(input)
    def decompose(self, task, max_subtasks=3):
        prompt = "You are a shrewd planner and are capable of reducing complex tasks into simpler ones (with a maximum of {max_subtasks} tasks).  However, you are also extremely busy and have no time to waste. Please decompose the task '{task}' into subtasks.  You must always return a comma-separated list of single-quoted strings, with nore more than {max_subtasks} elements. Each task must be self-contained such that it can be interpreted as a single request. Please return no other text, which would include numbering and newline characters. Use this format: <task1>, <task2>, <task3>.".format(task=task, max_subtasks=max_subtasks)
        user_input = [{"role": "user", "content": prompt}]
        # Query the model
        response = self.model.generate_response(messages=user_input)
        return response