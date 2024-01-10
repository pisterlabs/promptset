import json
import sys

sys.path.append("/Users/dtolley/Documents/Projects/gptretrieval/gptretrieval")
from services import openai

GPT_TOKEN_LENGTH = 4096
GPT_MODEL = "gpt-4"

labels_dict = {
    0: {
        "name": "Class or Struct Definition",
        "description": "Code that defines a class or struct. Excludes the methods within a class; only includes the class signature and member variables.",
    },
    1: {
        "name": "Function or Method Definition",
        "description": "Code that defines a function or method. Does not include usage examples of the function or method.",
    },
    2: {
        "name": "Code Usage or Example",
        "description": "Examples of how to use certain code, functions, or classes. Distinct from the actual definition of functions or classes.",
    },
    3: {
        "name": "Instructional Code Implementation",
        "description": "How to implement or use code.  Such as how do I create a function to do ABC, or how is a class used to do XYZ.",
    },
    4: {
        "name": "Database Implementation",
        "description": "Code that implements or calls a database.",
    },
    5: {
        "name": "Error Handling",
        "description": "Code segments dedicated to handling errors or exceptions.",
    },
    6: {
        "name": "UI Code",
        "description": "Code related to user interface design and interaction.",
    },
    7: {
        "name": "Configuration Code",
        "description": "Code used for configuring the system, application, or environment.",
    },
    8: {
        "name": "Documentation",
        "description": "Comments and documentation that explain the code. Does not include code itself.",
    },
    9: {
        "name": "REST API Implementation or Usage",
        "description": "Code that either implements a server or client, or calls a REST API.",
    },
    10: {
        "name": "Code Usage Search",
        "description": "Looking for location and or file where a specific function or class or variable is being used",
    },
}

test_cases = {
    0: [
        {
            "question": "How do you define a simple class in Python?",
            "code": "class MyClass:\n    def __init__(self, name):\n        self.name = name",
            "answer": (0, 0),  # Class or Struct Definition for both question and code
        }
    ],
    1: [
        {
            "question": "Can you provide a function that adds two numbers?",
            "code": "def add_numbers(a, b):\n    return a + b",
            "answer": (
                1,
                1,
            ),  # Function or Method Definition for both question and code
        }
    ],
    2: [
        {
            "question": "How do I use the add_numbers function?",
            "code": "result = add_numbers(3, 5)\nprint(result)",
            "answer": (2, 2),  # Code Usage or Example for both question and code
        }
    ],
    3: [
        {
            "question": "Can you show an implementation of the bubble sort algorithm?",
            "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",
            "answer": (3, 3),  # Algorithm Implementation for both question and code
        }
    ],
    4: [
        {
            "question": "How do you implement a stack data structure?",
            "code": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()",
            "answer": (
                4,
                4,
            ),  # Data Structure Implementation for both question and code
        }
    ],
    5: [
        {
            "question": "How do you use the pandas library to read a CSV file?",
            "code": "import pandas as pd\ndata = pd.read_csv('file.csv')",
            "answer": (5, 5),  # Library or Package Usage for both question and code
        }
    ],
    6: [
        {
            "question": "Can you show me how to handle a division by zero error?",
            "code": "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero!')",
            "answer": (6, 6),  # Error Handling for both question and code
        }
    ],
    7: [
        {
            "question": "How can I create a button in a Tkinter window?",
            "code": "from tkinter import Tk, Button\nroot = Tk()\nbutton = Button(root, text='Click Me')\nbutton.pack()\nroot.mainloop()",
            "answer": (7, 7),  # UI Code for both question and code
        }
    ],
    8: [
        {
            "question": "How do you set up a configuration file for logging in Python?",
            "code": "[loggers]\nkeys=root\n\n[logger_root]\nlevel=DEBUG\nhandlers=consoleHandler\n\n[handlers]\nkeys=consoleHandler\n\n[handler_consoleHandler]\nclass=StreamHandler\nlevel=DEBUG\nformatter=consoleFormatter\nargs=(sys.stdout,)",
            "answer": (8, 8),  # Configuration Code for both question and code
        }
    ],
    9: [
        {
            "question": "What is the purpose of documentation in code?",
            "code": "# This function adds two numbers\ndef add_numbers(a, b):\n    # Add the numbers and return the result\n    return a + b",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
    10: [
        {
            "question": "How do you set up a configuration file for logging in Python?",
            "code": "class ABC",
            "answer": (8, 8),  # Configuration Code for both question and code
        }
    ],
    11: [
        {
            "question": "What is the purpose of documentation in code?",
            "code": "def add_let(a, b):\n  return 'abc' + 'b'",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
    12: [
        {
            "question": "Where in the code base is the class ABC?",
            "code": "line 123 in file.py",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
    13: [
        {
            "question": "show me the definition for class ABC?",
            "code": "line 123 of file abciscool.py\ndef testit(): \n  x = ABC()\n print(x)",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
    13: [
        {
            "question": "show me the definition for class ABC?",
            "code": "line 123 of file abciscool.py\nclass ABC\n self.x = 1",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
    14: [
        {
            "question": "what file and line is class ABC defined?",
            "code": "line 123 of file abciscool.py\nclass ABC\n self.x = 1",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
    15: [
        {
            "question": "show me the definition for class ABC?",
            "code": "line 123 of file abciscool.py\nclass BA\n self.x = 1",
            "answer": (9, 9),  # Documentation for both question and code
        }
    ],
}


def create_prompt_for_gpt(labels_dict):
    """
    Convert a dictionary of labels into a text block for GPT prompt.

    Parameters:
    labels_dict (dict): A dictionary containing label indices as keys and another dictionary with 'name' and 'description' as values.

    Returns:
    str: A text block suitable for use as a GPT prompt.
    """
    prompt = "The following are example labels but are not exclusive:\n\n"
    for index, label_info in labels_dict.items():
        prompt += f"Label - {label_info['name']}:\n"
        prompt += f"{label_info['description']}\n\n"
    return prompt


def classify_question(question: str, labels: str):
    """Call OpenAI and summarize the function or class definition"""

    prompt_text = create_prompt_for_gpt(labels_dict)
    question = question[:GPT_TOKEN_LENGTH]

    system_message = {
        "role": "system",
        "content": f"{prompt_text}\nYou can ask me to classify a question, \
            and I will return a label for the question formatted as json. \
            formatted as {{'question': 'label']}}",
    }
    user_message = {
        "role": "user",
        "content": f"Classify the following: Question - {question}",
    }

    functions = [
        {
            "name": "classify_question",
            "description": "A function which takes in the label for question",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_label": {
                        "type": "string",
                        "description": "The label index assigned to the question",
                    }
                },
                "required": ["question_label"],
            },
        }
    ]

    resp = openai.get_chat_completion(
        [system_message, user_message],
        functions,
        function_call={"name": "classify_question"},
        model=GPT_MODEL,
    )

    return resp


def classify_code(code: str, question: str, question_label: str):
    """
    Call OpenAI to generate potential code labels based on the question and question label.

    Parameters:
    code (str): The code snippet.
    question (str): The question text.
    question_label (str): The label assigned to the question.

    Returns:
    Response from OpenAI API.
    """
    # Craft the system message with the labels dictionary and instruction
    prompt_text = create_prompt_for_gpt(labels_dict)
    code = code[:GPT_TOKEN_LENGTH]

    system_message = {
        "role": "system",
        "content": f"{prompt_text}\nGiven a question and its classification, you can ask me to classify a code snippet. \
            The classification of the code snippet is '1' if it should align with the context provided by the question and its classification else its 0. \
                Think of the code classification as the role the code plays in the context of answering the classified question. \
                    For example, if the question is asking for a class definition, but the code snippet is using a class without \
                        defining it, the code snippet should be classified as '0' or irrelevant.",
    }

    user_message = {
        "role": "user",
        "content": f"The question is: '{question}'. It is classified as: '{question_label}'. Given this context, how would you classify the following code snippet: {code}?",
    }

    # Define the function for the API call
    functions = [
        {
            "name": "classify_code",
            "description": "A function which takes in a code label",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_label": {
                        "type": "integer",
                        "description": "The label for the code",
                    }
                },
                "required": ["code_label"],
            },
        }
    ]

    # Make the API call to GPT-4 with the crafted messages and function
    resp = openai.get_chat_completion(
        [system_message, user_message],
        functions,
        function_call={"name": "classify_code"},
        model=GPT_MODEL,
    )

    return resp


# create a main entry point
def main():
    # print the summary for each test case
    for label, test_case in test_cases.items():
        for case in test_case:
            gpt_question_response = classify_question(case["question"], labels_dict)
            # print question and code, and what the answer is supposed to be
            print("--------------------------------")
            print(f"Question: {case['question']}")
            print(f"Code: {case['code']}")
            print("--------------------------------")

            # Assuming gpt_response contains the indices of the predicted labels
            # You might need to adjust this part based on the actual structure of gpt_response
            predicted_question_label = gpt_question_response["function_args"][
                "question_label"
            ]

            gpt_code_response = classify_code(
                case["code"], case["question"], predicted_question_label
            )

            predicted_code_label = gpt_code_response["function_args"]["code_label"]

            print(
                f"GPT Response: Question - {predicted_question_label}, Code - {predicted_code_label}\n"
            )


if __name__ == "__main__":
    main()
