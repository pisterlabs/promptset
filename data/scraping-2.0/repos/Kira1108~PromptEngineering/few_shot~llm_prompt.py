from langchain.prompts import FewShotPromptTemplate
from llm_examples import examples, example_prompt

# this is the introduction to the problem
prefix = """
This is a very import problem to solve, you should be able to solve it correctly.
"""

# receive the user input
suffix = """
"question": "{input_question}",
"answer": 
"""

fewshot = FewShotPromptTemplate(
    prefix = prefix,
    examples = examples,
    example_prompt=example_prompt,
    suffix = suffix,
    input_variables=['input_question'],
    example_separator= "\n"
    
)

if __name__ == "__main__":
    print(fewshot.format(input_question = "Hahahahahah"))
