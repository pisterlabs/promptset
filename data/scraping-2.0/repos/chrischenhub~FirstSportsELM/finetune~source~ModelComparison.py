import json
import sys
from openai import OpenAI
import pandas as pd

def QuestionList(path = '../data/SportsQuestions.json'):
    question_path = path

    with open(question_path, 'r') as file:
        questions = json.load(file)

    return questions

def RubricList(path = '../data/grading_rubric.json'):
    file_path = path

    with open(file_path, 'r') as file:
        grading_rubric_data = json.load(file)

    # Converting the data into a list format
    rubric_list = []
    for category, scores in grading_rubric_data.items():
        rubric_text = f"{category}: " + "; ".join([f"Score {score}: {description}" for score, description in scores.items()])
        rubric_list.append(rubric_text)

    return rubric_list

def RateQueryResponse(client, question, rubric, a1, a2, a3, a4):
    client = OpenAI(api_key = client)

    response = client.chat.completions.create(
    model = "gpt-3.5-turbo-1106",
    messages = [
        {"role": "system", 
        "content": f"""You are an AI assistant assigned to grade ChatBot responses according to a provided rubric. \
                 Four responses will be presented, all attempting to answer the question: '{question}'. \
                 Evaluate responses using the rubric: '{rubric}'. Score these responses on a scale of 5. \
                 Only output score. Strictly format the scores as: '\d+, \d+, \d+, \d+'."""}, #context
        
        {"role": "user", 
        "content": f"""Please evaluate these responses solely based on the specified criteria in the rubric: 
        1. {a1}
        2. {a2}
        3. {a3}
        4. {a4}"""}  #prompt
        ],
        temperature = 0.1
        )

    ResponseStr = response.choices[0].message.content.strip()

    return ResponseStr

def LoadAnswer(p1 = '../data/LlaMa2Answer.json',
               p2 = '../data/DefaultAnswer.json',
               p3 = '../data/RandomChatBotAnswer.json',
               p4 = '../data/TargetAnswer.json'):

    with open(p1, 'r') as file1:
        a1 = json.load(file1)

    with open(p2, 'r') as file2:
        a2 = json.load(file2)
    
    with open(p3, 'r') as file3:
        a3 = json.load(file3)
    
    with open(p4, 'r') as file4:
        a4 = json.load(file4)

    return a1,a2,a3,a4

def Compare(client):
    Rubrics = RubricList()
    Questions = QuestionList()
    A1, A2, A3, A4 = LoadAnswer()

    for index, rubric in enumerate(Rubrics):
        all_results = []

        for question, a1, a2, a3, a4 in zip(Questions, A1, A2, A3, A4):
            Response = RateQueryResponse(client, question, rubric, a1, a2, a3, a4)
            number_list = [int(x) for x in Response.split(',')]
            print(number_list)

            all_results.append(number_list)

        df = pd.DataFrame(all_results, columns = ['LlaMa2', 'Default', 'Random', 'Target'])
        file_path = f"../data/rubric_{index}.csv"
        df.to_csv(file_path, index=False)

if __name__ == '__main__': 
    client = sys.argv[1]
    Compare(client)