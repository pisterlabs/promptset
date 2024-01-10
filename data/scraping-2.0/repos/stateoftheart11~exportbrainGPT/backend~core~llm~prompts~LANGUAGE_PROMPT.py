from langchain.prompts.prompt import PromptTemplate
from models.brains import Personality
from llm.prompts.personality_description_list import personality_description_list

default_prompt_template = """Your are an expert. A person will ask you a question and you will provide a helpful answer. Write the answer in the same language as the question. If you don't know the answer, just say that you don't know. Don't try to make up an answer. Use the following context to answer the question:


{context}

Question: {question}
Helpful Answer:"""
DEFAULT_QA_PROMPT = PromptTemplate(
    template=default_prompt_template, input_variables=["context", "question"]
)


prompt_templete = """You are an Expert, Your figurative expression is: {description}. You should answer match your personality. Use the following pieces of context to answer the question at the end. If context is empty, ignore context.
These contexts may vary, from the your profile or experiences to more informative elements. Additionally, these contexts could encompass knowledge your possesses.
Contexts:


{context}

Question: {question}
Your Answer: 
"""

def qa_prompt(personality:Personality = None):

    if personality:
        description = personality_description_list[personality.extraversion * 16 + personality.neuroticism * 4 + personality.conscientiousness]

        personal_prompt_template = prompt_templete.replace("{description}", description)
        prompt = PromptTemplate(
            template=personal_prompt_template, input_variables=["context", "question"]
        )
        return prompt
    else:
        return DEFAULT_QA_PROMPT