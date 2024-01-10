# This code brings best questions for interview

# load modules
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json


# get api key from ApiKey.txt
apiKey = ""
with open("ApiKey.txt",'r') as f:
    apiKey = f.readline()
    apiKey = apiKey.strip()

def get_questions(role, experience):
    # create model to predict
    llm = OpenAI(openai_api_key=apiKey)
    # Prompt template for getting questions
    prompt_search_questions = PromptTemplate.from_template("Provide minimum 15 interview questions for {role} for {experience} candidate?")
    # format template to final prompt
    questions = prompt_search_questions.format(role = role, experience = experience)
    # Getting output of prompt 
    questions_output = llm.predict(questions)

    print(questions_output)

    with open("question_response.txt",'w') as f:
        f.write(questions_output)

    # convert questions from string to list
    questions_output = questions_output.strip()
    questions_output = questions_output.split("\n")
    questions_lst = []
    for i in questions_output:
        i = i.strip()
        try:
            i = i.split('. ')
            questions_lst.append(i[1])
        except:
            pass

    # write questions in josn file
    with open("questions.json",'w') as f:
        f.write(json.dumps(questions_lst))
    
    return questions_lst



role = "Frontend Developer"
experience = "fresher"

output = get_questions(role,experience)

print(output)