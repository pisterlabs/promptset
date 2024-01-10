# This code takes question and answer as input and provide its rating out of 10

# load modules
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import json

# get api key from ApiKey.txt
apiKey = ""
with open("ApiKey.txt",'r') as f:
    apiKey = f.readline()
    apiKey = apiKey.strip()

def score_answer(role, experience, question, answer):
    # create model to predict
    chat = ChatOpenAI(openai_api_key=apiKey)

    # Prompt templates to get score of answer
    system_template = (
        '''
        You are an interviewer, taking interview for {role} of a {experience} candidate.
        Question : {question}
        '''
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "Answer : {answer}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    system_template_1 = "Rate this answer from 0 to 10 and provide score and feedback in json format."
    system_message_prompt_1 = SystemMessagePromptTemplate.from_template(system_template_1)

    # create final chat prompt 
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt, system_message_prompt_1]
    )

    # Getting output of prompt 
    score = chat(
        chat_prompt.format_prompt(
            role = role, experience=experience, question=question, answer = answer
        ).to_messages()
    )

    return json.loads(score.content)



# variable to store values needed to find score
role = "Python developer"
experience = "fresher"
question = "How familiar are you with unit testing frameworks in Python?"
answer = "Avoid using recursion: Recursive functions can slow down your code because they take up a lot of memory. Instead, use iteration. Use NumPy and SciPy: NumPy and SciPy are powerful libraries that can help you optimize your code for scientific and mathematical computing. Use Cython to speed up critical parts of the code."


output = score_answer(role,experience,question,answer)

print(output)