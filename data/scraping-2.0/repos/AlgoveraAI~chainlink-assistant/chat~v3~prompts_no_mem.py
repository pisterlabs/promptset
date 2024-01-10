from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

potential_answer_system_template = """
As an AI assistant, your task is to locate the segment in a document that provides the answer to a user's inquiry about Chainlink. 
If the document doesn't contain the required information, respond with 'no answer'. 
Ensure to return only the segment containing the precise answer.
Ensure your report is in MARKDOWN format.
Add source of the document at the end of the answer.
"""

potential_answer_human_template = """
User's question: {question}

Document: {document}

Answer:
Source:
"""

POTENTIAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(potential_answer_system_template),
        HumanMessagePromptTemplate.from_template(potential_answer_human_template),
    ]
)

final_answer_system_template = """
As an AI assistant helping answer a user's question about Chainlink, your task is to provide the answer to the user's question based on the documents provided. 
If the document doesn't contain the required information, respond with 'I don't know'.
Each point in your answer should be formatted with corresponding reference(s) using markdown. Conclude your response with a footnote that enumerates all the references involved. 
The footnote should be formatted as follows: 
```
References:
[^1^]: <reference 1> 
[^2^]: <reference 2> 
[^3^]: <reference 3>
```
"""

final_answer_human_template = """
User's question: {question}

Document: {document}

Answer:
"""

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(final_answer_system_template),
        HumanMessagePromptTemplate.from_template(final_answer_human_template),
    ]
)

verification_system_template = """
You are an AI assistant.
You are helping a user find information about Chainlink.
Given the a question and an answer pair, please verify if the answer is correct.
If the answer is "the document does not contain the answer", please answer "no".
Please ONLY answer yes or no.
"""
verification_human_template = """
Question: {question}
Answer: {answer}
"""

VERIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(verification_system_template),
        HumanMessagePromptTemplate.from_template(verification_human_template),
    ]
)

final_answer_system_template_2 = """
As an AI assistant helping answer a user's question about Chainlink, your task is to provide the answer to the user's question based on the documents provided.
If the document doesn't contain the required information, respond with 'I don't know'.
Each point in your answer should be formatted with corresponding reference(s) using markdown. Conclude your response with a footnote that enumerates all the references involved.
The footnote should be formatted as follows: 
```
References:
[^1^]: <reference 1> 
[^2^]: <reference 2> 
[^3^]: <reference 3>
```
"""

final_answer_human_template_2 = """
User's question: {question}

Document: {document}

Answer:
"""

FINAL_ANSWER_PROMPT_2 = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(final_answer_system_template_2),
        HumanMessagePromptTemplate.from_template(final_answer_human_template_2),
    ]
)
