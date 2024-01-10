
#from __future__ import print_function
from oauth2client import client, file, tools
from apiclient import discovery
from httplib2 import Http
import openai
import re

########################################## GENERATE THE QUIZ ############################################
             #1-  interacting with the OpenAI API.
                 #2-  generating a quiz based on the provided text.
                      #3-  Regular expressions are used to extract different components from the content
              #####################################################

# sets the value of the api_key attribute of the openai object
openai.api_key = "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"

# Link to OPEN AI documentation : https://platform.openai.com/docs/quickstart/build-your-application.

def Quiz_Generator(User_text):
    responce = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": User_text} ])

    # assigns the content of the message to the variable result .
    result = responce.choices[0].message.content

    # Regular Expression: https://www.mygreatlearning.com/blog/regular-expression-in-python/
    quest = re.compile(r'^\d+(?:\)|\.|\-)(.+\?$)')
    opt = re.compile(r'^[a-zA-Z](?:\)|\.|\-)(.+$)')
    ans= re.compile(r'Answer:\s[a-zA-Z](?:\)|\.|\-)(.+$)')
    
    questions = []
    options=[]
    sub =[]
    answers =[]

    for line in result.splitlines():
        if line == '':
            if sub:
                options.append(sub)
                sub=[]
        else:
            if quest.match(line):
                line_mod = line.strip()
                questions.append(line_mod)
            if opt.match(line):
                line_mod = line.strip()
                sub.append(line_mod)
            if ans.match(line):
                line_mod= line.strip()
                answers.append(line_mod)
    if sub:
        options.append(sub)




##################################################################################
        #################### CREATE THE GOOGLE FORM######################
                    ##########################################

    store = file.Storage('client.json')
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_google_form_secrets.json',  "https://www.googleapis.com/auth/forms.body")
        creds = tools.run_flow(flow, store)

    form_service = discovery.build('forms', 'v1', http=creds.authorize(
        Http()), discoveryServiceUrl="https://forms.googleapis.com/$discovery/rest?version=v1", static_discovery=False)

    # Request body for creating a form
    FORM_body = {
        "info": {
            "title": "Quiz form",
        }
    }
    # Creates the initial form
    result = form_service.forms().create(body=FORM_body).execute()
    # Request body to add a multiple-choice question
    # JSON to convert the form into a quiz
    update = {
        "requests": [
            {
                "updateSettings": {
                    "settings": {
                        "quizSettings": {
                            "isQuiz": True
                        }
                    },
                    "updateMask": "quizSettings.isQuiz"
                }
            }
        ]
    }
    # Converts the form into a quiz
    question_setting = form_service.forms().batchUpdate(formId=result["formId"],body=update).execute()
    for i in range(len(questions)): 
        NEW_QUESTION = {
            "requests": [{
                "createItem": {
                    "item": {
                        "title": questions[i],
                        "questionItem": {
                            "question": {
                                "required": True,
                                "choiceQuestion": {
                                    "type": "RADIO",
                                    "options": [{"value":j} for j in options[i]],
                                    "shuffle": True
                                }
                            }
                        },
                    },
                    "location": {
                        "index": i
                    }
                }
            }]
        }
        question_setting = form_service.forms().batchUpdate(formId=result["formId"], body=NEW_QUESTION).execute()

    get_result = form_service.forms().get(formId=result["formId"]).execute()
    return get_result['responderUri']
