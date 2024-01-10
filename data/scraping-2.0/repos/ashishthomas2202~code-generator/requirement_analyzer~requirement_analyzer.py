import os
import openai
import ast


def analyze(project):

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # output_format = {
    #     "questions": ["question 1", "question 2", "question 3"]
    # }
    # Updated prompt with more specific instructions
    # prompt = f"Translate the user requirements: {requirements}, and generate additional business related questions required for software development based on these requirements.Note: The questions should be formatted as a Python dictionary with attributes and key value pair in double quotes like {output_format}"
    # prompt = f"As a software development firm, we aim to better understand your requirements for the project related to {project}. Please provide us with additional insights and details that will help us tailor our approach to meet your business needs. We will frame our questions in a way that is easy to understand for non-technical clients.  Please generate 5 questions. if its a technical question, please provide some recommended one word answers. here is the output format for the questions: {output_format}."
    # response = createResponse(prompt)
    questions = generateQuestions(project)

    # print("response", response)
    # questions = ast.literal_eval(response)['questions']

    # print("parsed questions", questions)

    detailed_requirements = []

    for index, question in enumerate(questions):

        done = False
        while not done:
            answer = input(f"{index + 1}. {question}: ")
            if len(answer) > 0:
                detailed_requirements.append({
                    "question": question,
                    "answer": answer
                })
                done = True

    # project["detailed_requirements"] = detailed_requirements
    # return detailed_requirements
    # output_format = {
    #     "summary": {
    #         "topic-name": "summary",

    #     }
    # }

    # response = createResponse(
    #     f"Take the following requirements {detailed_requirements} and convert it into project requirements. the requirements must be in form of {output_format}. the requirements must be very detailed. use key in kebab case, descriptive and short. also the json must have the root key as 'summary'")

    # summarize_response = ast.literal_eval(response)['summary']

    requirements = generateRequirements(detailed_requirements, project)
    return requirements


def generateQuestions(project):
    output_format = {
        "questions": ["question 1", "question 2", "question 3"]
    }

    messages = [
        {
            "role": "system",
            "content": f"""The purpose of this model is to generate questions based on the user requirements. 
                        These question should be based on the development needs like backend frameworks and frontend 
                        frameworks or about business, features. The questions should be formatted as a Python dictionary 
                        with attributes and key value pair in double quotes like {output_format}.Make sure the output 
                        dictionary has the root key as 'questions'."""
        },
        {
            "role": "user",
            "content": f"Here are the requirements: {project}"
        }
    ]

    response = createResponse(messages)
    questions = []
    try:
        questions = ast.literal_eval(response)['questions']
    except KeyError as e:
        print("error", e)
        questions = generateQuestions(project)

    return questions


def generateRequirements(detailed_requirements, project):
    output_format = {
        "requirements": {
            "web": {
                "frontend_frameworks": ["framework 1", "framework 2"],
                "backend_frameworks": ["framework 1", "framework 2"],
                "features": ["feature 1", "feature 2"],
            },
            "mobile": {
                "frontend_frameworks": ["framework 1", "framework 2"],
                "backend_frameworks": ["framework 1", "framework 2"],
                "features": ["feature 1", "feature 2"],
            }
        }
    }

    messages = [
        {
            "role": "system",
            "content": f"""The purpose of this model is to generate project requirements based on the question answers. 
                            breakdown the answers and make more detailed requirements, breakdown general answers to technical elements. 
                            the requirements must be in form of {output_format}. the requirements must be very detailed.
                            use key in kebab case, descriptive and short(one word if possible). also the python dictionary
                            must have the root key as'requirements'. no other text must be present in the output besides dictionary."""
        },
        {
            "role": "user",
            "content": f"Here are the question answers: {detailed_requirements}"
        }
    ]

    response = createResponse(messages)

    try:
        requirements = ast.literal_eval(response)['requirements']
    except KeyError as e:
        print("error", e)
        requirements = generateRequirements(detailed_requirements, project)

    return requirements


def createResponse(messages):
    # response = openai.Completion.create(
    #     # model="text-davinci-003",
    #     model="gpt-3.5-turbo-instruct",
    #     prompt=prompt,
    #     temperature=1,  # Adjust temperature for desired creativity
    #     max_tokens=3800,    # Adjust max_tokens based on desired response length
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )
    # return response.choices[0].text

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=1,
        max_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response)
    return response.choices[0].message.content
