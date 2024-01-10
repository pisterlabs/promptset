#%%
import sys
from openai import OpenAI
import time
import json
import re

#%%
# Function to generate tags
def TagGeneration(client, NumTag = 200):
    response = client.chat.completions.create(
    model = "gpt-4",
    messages = [
        {"role": "system", 
        "content": """You are an intelligent assistant tasked with generating a list of daily sports-related tags. 
        These tags should cater to the interests of sports fan, focusing most frequently asked topics on popular sports, 
        sports news, professional sports teams, updates on major sports leagues and events, profiles of prominent athletes,
        and insights into emerging sports figures, and easy-to-understand strategies and techniques. 
        These tags will be used for fine-tuning a language learning model, aiming to provide engaging, informative, 
        and relatable sports content for sports fans. Focus on generating tags that can facilitate a wide range 
        of sports-related queries and discussions, providing a solid foundation for creating a rich and varied 
        Q&A dataset. Each tag should start on a new line and be preceded by its number and a period. 
        Organize it as follows: '1. [First tag] 2. [Second tag] ...'"""}, #context
        
        {"role": "user", 
        "content": f"""Please generate a list of {NumTag} unique sports-related tags."""}  #prompt
    ])

    ResponseStr = response.choices[0].message.content.strip()

    return ResponseStr

# Convert tags into list
def Tag2List(ResponseStr):
    lines = ResponseStr.strip().split('\n')

    SportsList = [line.split('. ', 1)[1] for line in lines if line]

    return SportsList

# Function to generate questions from tag
def QuestionGeneration(client, tag, NumQuestion):
    response = client.chat.completions.create(
        model = "gpt-4",
        messages = [
            {"role": "system", 
             "content": f"""Generate {NumQuestion} questions that a user might naturally ask in a casual conversation 
             about {tag}. Each question should be engaging, easy to understand for a broad audience, distinct from 
             each other, and sport-related. Instead of asking most up-to-date questions, focus more on sports topics whenever before 
             2023. Additionally, ensure that the questions are asked in diverse ways. Ensure a mix of question 
             types, such as list-based, specific inquiries, individual recognitions, quantifying, and comparative questions, 
             while keeping them engaging and easy to understand. Format the {NumQuestion} questions as a numbered list starting 
             from 1. Each question should start on a new line and be preceded by its number and a period. Organize it as follows: 
             '(1). [First question] (2). [Second question]'"""},

            {"role": "user", 
             "content": f"Give me a list of {NumQuestion} questions"}
        ]
    )
    return response.choices[0].message.content.strip()

# Function to paraphrase a question
def QuestionParaphrase(client, question, NumParaphrase):
    ParaQuestions = []
    for _ in range(NumParaphrase):
        response = client.chat.completions.create(
            model = "gpt-4",
            messages = [
                {"role": "system", 
                 "content": f"""Rewrite the following question in a 
                 different way: {question}"""},
                
                {"role": "user", "content": ""}
            ]
        )
        ParaQuestions.append(response.choices[0].message.content.strip())
        time.sleep(1)  
    return ParaQuestions

# Function to generate answers for a question
def AnswerGeneration(client, question, NumAnswer):
    answers = []

    response = client.chat.completions.create(
        model = "gpt-4",
        messages = [
            {"role": "system", 
             "content": f"""Provide {NumAnswer} different answers to the following question, each with a slightly 
                different style. Ensure that each answer is a somehow different from the other or have different 
                information. Format the answers as a numbered list starting from 1. Each answer should start on a 
                new line and be preceded by its number and a period. 
                Organize it as follows: '(1). [First answer] (2). [Second answer]': {question}"""},

            {"role": "user", 
             "content": f"Give me a list of {NumAnswer} answers"}
        ]
    )
    answers = response.choices[0].message.content.strip()
    return answers

# Function to pipeline entire process
def Tag2All(client, tags, NumQuestion, NumParaphrase, NumAnswer):
    finetune = []

    for index, tag in enumerate(tags):
        print(f"Processing tag {index + 1}/{len(tags)}: {tag}")

        questions = QuestionGeneration(client, tag, NumQuestion)
        QuestionList = questions.strip().split('\n')
        QuestionList = [line.split('. ', 1)[1] for line in QuestionList if line]

        for question in QuestionList:
            ParaQuestions = QuestionParaphrase(client, question, NumParaphrase)
            AllQuestions = [question] + ParaQuestions

            answers = AnswerGeneration(client, question, NumAnswer)
            answers = re.split(r'\(\d+\)\.', answers)
            answers = [answer.strip() for answer in answers if answer.strip()]

            intent = {"tag": tag,
                      "questions": AllQuestions,
                      "responses": answers}

            finetune.append(intent)

    JsonData = json.dumps({"finetune": finetune}, indent = 4)

    return JsonData


def JsonStore(dir, JsonData):
    with open(dir, 'w') as file:
        file.write(JsonData)


#%%
if __name__ == '__main__': 
    api_key = sys.argv[1]

    NumTag = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    NumQuestion = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    NumParaphrase = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    NumAnswer = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    client = OpenAI(api_key = api_key)
    TagList = Tag2List(TagGeneration(client, NumTag))

    JsonData = Tag2All(client, TagList, NumQuestion, NumParaphrase, NumAnswer)

    JsonStore('../data/GPTGenerated.json', JsonData)
