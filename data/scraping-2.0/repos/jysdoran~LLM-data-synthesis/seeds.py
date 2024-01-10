""" Generate a collection of seed concepts to promote the uniqueness of generated data"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from tqdm import tqdm

from generate import OPENAI_KEY, tokenize_docstring_from_string
from data import JSONLDataset
from tokenise_fp import function_data_from_string

subconcept_template = """{description}
Generate a list of {k} subconcepts of "{concept}" and a short description of how each relates to {concept}.
Format each list entry as: x. Subconcept: Description."""
SUBCONCEPT_PROMPT = HumanMessagePromptTemplate.from_template(subconcept_template)

INITIAL_SEEDS = [
    "python",
    "software engineering",
    "algorithms",
    "data structures",
    "software design patterns",
]


@dataclass
class Concept:
    name: str
    description: str
    subconcepts: List["Concept"] = None

    def to_dict(self):
        if not self.subconcepts:
            return {
                "name": self.name,
                "description": self.description,
            }
        else:
            return {
                "name": self.name,
                "description": self.description,
                "subconcepts": [c.to_dict() for c in self.subconcepts],
            }

    @staticmethod
    def from_dict(d):
        if "subconcepts" in d:
            return Concept(
                name=d["name"],
                description=d["description"],
                subconcepts=[Concept.from_dict(c) for c in d["subconcepts"]],
            )
        else:
            return Concept(
                name=d["name"],
                description=d["description"],
            )

    @property
    def unique_names(self):
        names = {self.name}
        if self.subconcepts:
            for subconcept in self.subconcepts:
                names.update(subconcept.unique_names)
        return names

    @property
    def flat(self):
        concepts = [self]
        if self.subconcepts:
            for subconcept in self.subconcepts:
                concepts.extend(subconcept.flat)
        return concepts

    @property
    def leaves(self):
        if not self.subconcepts:
            return [self]
        concepts = []
        for subconcept in self.subconcepts:
            concepts.extend(subconcept.leaves)
        return concepts


coding_concepts = [
    Concept(
        name="Variables",
        description="Variables are used to store and manipulate data within a program. They are essential for performing calculations, storing user input, and maintaining state throughout the execution of the code.",
        subconcepts=None,
    ),
    Concept(
        name="Control Flow",
        description="Control flow refers to the order in which statements and instructions are executed within a program. Concepts such as loops (e.g., for, while) and conditional statements (e.g., if, else) allow programmers to control the flow of execution based on certain conditions.",
        subconcepts=None,
    ),
    Concept(
        name="Functions",
        description="Functions are blocks of reusable code that perform specific tasks. They enable programmers to modularize their code, improve code organization, and promote code reuse. Functions take inputs (parameters) and can return outputs.",
        subconcepts=None,
    ),
    Concept(
        name="Data Structures",
        description="Data structures are containers used to organize and manage data efficiently. Examples include arrays, linked lists, stacks, queues, and trees. Understanding different data structures is crucial for storing, accessing, and manipulating data effectively.",
        subconcepts=None,
    ),
    Concept(
        name="Algorithms",
        description="Algorithms are step-by-step procedures or instructions for solving problems. They provide a logical approach to accomplish specific tasks efficiently. Knowledge of algorithms helps in optimizing code performance and finding the most effective solutions.",
        subconcepts=None,
    ),
    Concept(
        name="Object-Oriented Programming (OOP)",
        description="OOP is a programming paradigm that structures code around objects, which represent real-world entities. It emphasizes encapsulation, inheritance, and polymorphism. OOP allows for code organization, modularity, and reusability.",
        subconcepts=None,
    ),
    Concept(
        name="Debugging",
        description="Debugging is the process of identifying and fixing errors (bugs) in code. It involves techniques such as analyzing error messages, using breakpoints, and stepping through code to locate and resolve issues. Debugging is crucial for ensuring code correctness.",
        subconcepts=None,
    ),
    Concept(
        name="Libraries and APIs",
        description="Libraries and APIs (Application Programming Interfaces) provide pre-written code and functionalities that programmers can use in their projects. They offer ready-made solutions for common tasks, allowing developers to save time and effort.",
        subconcepts=None,
    ),
    Concept(
        name="Databases",
        description="Databases are structured collections of data that provide persistent storage. Knowledge of databases and related concepts, such as querying languages (e.g., SQL), allows programmers to store, retrieve, and manipulate data efficiently in their applications.",
        subconcepts=None,
    ),
    Concept(
        name="Version Control",
        description="Version control systems (e.g., Git) help manage changes to code over time. They allow programmers to track revisions, collaborate with others, and easily revert to previous versions if needed. Version control promotes code stability and facilitates teamwork.",
        subconcepts=None,
    ),
]


def parse_response_concept(message):
    # Splitting the text list into a Python list using regular expression
    matches = re.findall(r"\d+[).] (.+)(?=: )([^\n]+)(?:\n|$)", message)

    return [Concept(*match) for match in matches]


def generate_subconcepts(chat_model: ChatOpenAI, concept: Concept, k=10):
    # Take a seed concept and generate a list of subconcepts
    prompt = ChatPromptTemplate.from_messages([SUBCONCEPT_PROMPT])

    try:
        message = chat_model.predict_messages(
            messages=prompt.format_messages(
                concept=concept.name, description=concept.description, k=k
            ),
            max_tokens=1000,
        )
        subconcepts = parse_response_concept(message.content)
        if len(subconcepts) < k:
            raise ValueError("Not enough subconcepts parsed from message: ", message)

    except Exception as e:
        print(e)
        return []

    return subconcepts


def generate_concept_tree(chat_model, concept, child_counts: List[int]):
    if child_counts:
        if not concept.subconcepts or len(concept.subconcepts) < child_counts[0]:
            print("Retrieving missing subconcepts for ", concept.name)
            concept.subconcepts = generate_subconcepts(
                chat_model, concept, k=child_counts[0]
            )
        # elif :
        #     concept.subconcepts.extend(
        #         generate_subconcepts(
        #             chat_model, concept, k=child_counts[0] - len(concept.subconcepts)
        #         )
        #     )
        for subconcept in concept.subconcepts:
            generate_concept_tree(chat_model, subconcept, child_counts[1:])


def generate_concepts(chat_model):
    # parent_concept = Concept("Coding", "Programming", subconcepts=coding_concepts)
    with open("coding_concepts.json", "r") as f:
        parent_concept = Concept.from_dict(json.load(f))
    for concept in tqdm(parent_concept.subconcepts):
        generate_concept_tree(chat_model, concept, [10, 16])
    with open("coding_concepts.json", "w") as f:
        json.dump(parent_concept.to_dict(), f)


def parse_response_code(message):
    # Split by function number to get individual function descriptions
    functions = re.split(r"\d+\.\s*Function:", message)[
        1:
    ]  # Skip the first split since it contains unrelated content

    data = []

    for func in functions:
        function_dict = {}

        # Extract function name
        function_name = re.search(r"(\w+)", func).group(1)
        function_dict["func_name"] = function_name

        # Extract relation to the concept of Variables
        relation_match = re.search(
            r"Relation to the concept of [^:]*: ([\s\S]+?)(?=(Docstring:|Code:))", func
        )
        if relation_match:
            function_dict["relation"] = relation_match.group(1).strip()

        # Extract docstring
        docstring_match = re.search(r"Docstring: ([\s\S]+?)(?=Code:)", func)
        if docstring_match:
            function_dict["docstring"] = docstring_match.group(1).strip()

        # Extract code
        code_match = re.search(r"Code:[\s\n]*(?:```)?(?:python)?([^`]+)(?:(```)|$)", func)
        if code_match:
            function_dict["code"] = code_match.group(1).strip()
        else:
            print("No code found for list item: ", func)

        data.append(function_dict)

    return data


def generate_concept_data(chat_model, concepts, k=10):
    data_template = """Concept: {description}
Generate the code and docstring for {k} short python functions that effectively demonstrate the concept of {concept}.
Each list item should be in the following format:
x.  Function:
    Relation to the concept of {concept}:
    Docstring:
    Code:"""
    data_prompt = HumanMessagePromptTemplate.from_template(data_template)
    chat_prompt = ChatPromptTemplate.from_messages([data_prompt])

    data = []

    for concept in tqdm(concepts):
        try:
            message = chat_model.predict_messages(
                chat_prompt.format_messages(
                    concept=concept.name, description=concept.description, k=k
                ),
                max_tokens=3000,
            )

            functions = parse_response_code(message.content)
            if len(functions) < k:
                raise ValueError("Not enough functions parsed from message: ", message)

        except Exception as e:
            print(e)
            functions = [{} for _ in range(k)]

        data.extend(functions)

    return data

def validate_concept_tree(concept_tree):
    print("Total", len(concept_tree.leaves))
    print("Unique names", len(concept_tree.unique_names))

    layer = [concept_tree]
    while layer:
        print(len(layer))
        new_layer = []
        lens = []
        for concept in layer:
            if concept.subconcepts:
                lens.append(len(concept.subconcepts))
                new_layer.extend(concept.subconcepts)
        print(lens)
        layer = new_layer

def fix_concept_data(chat_model, concept_tree):
    leaves = concept_tree.leaves

    concept_data = JSONLDataset("./datasets/subconcepts_8_synthetic_unprocessed.jsonl")

    missing_code = defaultdict(list)
    for i, data in enumerate(concept_data):
        if not data.get("code") or not data.get("docstring"):
            missing_code[i//8].append(i)
        else:
            try:
                # Attempt to parse the function from the code
                data = function_data_from_string(data["code"])
                if not data.get("function_tokens"):
                    raise ValueError("No tokens found in examples", data)
            except ValueError:
                print(data)
                missing_code[i//8].append(i)

    print(len(missing_code), missing_code)

    for leaf_i, data_idxs in missing_code.items():
        # Generate code for each missing data point
        data = generate_concept_data(chat_model, [leaves[leaf_i]], k=len(data_idxs))
        for i, d in zip(data_idxs, data):
            concept_data[i] = d

    concept_data.save_jsonl()

def process_jsonl():
    data = JSONLDataset("./datasets/subconcepts_8_synthetic_unprocessed.jsonl")
    print(len(data))
    errors = []
    for i, d in enumerate(data):
        try:
            docstring = d["docstring"]
            function_data = function_data_from_string(d["code"])
            d.update(function_data)
            d["docstring"] = docstring
            d["docstring_summary"] = docstring
            d["docstring_tokens"] = tokenize_docstring_from_string(docstring)
            if not d.get("docstring_tokens") or not d.get("function_tokens"):
                raise ValueError("No tokens found in examples", d)
        except Exception as e:
            errors.append(i)
            print(e)
    print(len(errors), errors)
    if not errors:
        data.save_jsonl("./datasets/subconcepts_8_synthetic.jsonl")

def main():
    chat_gpt = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name="gpt-3.5-turbo")
    # generate_concepts(chat_gpt)
    # validate_concept_tree()

    with open("coding_concepts.json", "r") as f:
        concept_tree = Concept.from_dict(json.load(f))

    # k = 8
    # data = JSONLDataset()
    # data.extend(generate_concept_data(chat_gpt, all_concepts.leaves, k=k))
    # data.save_jsonl(f"./datasets/subconcepts_{k}_synthetic.jsonl")
    # fix_concept_data(chat_gpt, concept_tree)
    process_jsonl()
    # generate_concept_data(chat_gpt, coding_concepts, k=2)



if __name__ == "__main__":
    main()
