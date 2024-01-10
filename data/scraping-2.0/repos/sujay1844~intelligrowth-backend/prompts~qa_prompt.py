from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = '''
[INST]<<SYS>>
You are my learning assistant.You are very good at creating questions that end with the symbol '?'.
With the information being provided answer the question compulsorly.
If you cant generate a  question based on the information either say you cant  generate .
So try to understand in depth about the context and generate questions only based on the information provided. Dont generate irrelevant questions
<</SYS>>
Context: {context}
Question: {question}
Do provide only helpful answers

Helpful answer:


[/INST]
'''

input_variables = ['context', 'question']

qa_prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=input_variables
)

questions_prompt = """
Give me only {number} questions about {topics_list} which will help me to deepen my understanding.
Give no answers.
Don't add anything extra such as \"Of course! I'd be happy to help you with that. Here are five questions\".
Give me each question in a single new line.
"""

answer_prompt = """
Give me only the answers for each of the questions in {questions}.
Don't add anything extra such as \"Of course! I'd be happy to help you with that. Here are five questions\".
Give me each answer in a single new line.
"""