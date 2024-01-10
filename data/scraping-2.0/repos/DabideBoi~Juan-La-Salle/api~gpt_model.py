import openai
OPENAI_API_KEY = 'key here please'
openai.api_key = OPENAI_API_KEY
#Reliable and Enhanced Messaging Intelligence
def REMI(msg):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    max_tokens=250,
    temperature=1.2,
    messages=[
            {"role": "system", "content": "You are ReMI. Your goal is to help Juan, our chatbot sound more human. Rephrase Juan's message reply without explaination."},
            {"role": "user", "content": msg}
        ])
    print(msg)
    reply = response['choices'][0]['message']['content']
    print(reply)
    return reply
# Spam Protection And MItigation
def SPAMI(msg):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    max_tokens=10,
    temperature=1.2,
    messages=[
            {"role": "system", "content": "You will return whether text given is considered useless. You will return '[True]' or '[False]' only without any explanation. This is the text '"+ msg +"'"},
        ])
    print(msg)
    reply = response['choices'][0]['message']['content']
    print(reply)
    return reply
#CLAsification System Integration
def CLASI(msg):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    max_tokens=10,
    temperature=1.2,
    messages=[
            {"role": "system", "content": """
                Instruction: Categorize the text to one available value. Only choose from the values given. Do not answer with any explanation or note. The available values are:
                greeting
                Your Name
                General Kenobi
                noanswer 
                PreReg
                reserve
                regular student
                shift
                where grades?
                goodbye
                thanks
                pre-registration
                reservation
                regular student
                shift
                gone
                access registration
                program code and section code
                class standing
                transfer
                officially enrolled
                transaction 2
                online enrollment limit
                cannot add subject
                program code and section
                name update
                graduation application
                graduation payments and fee
                pay w/o attendance
                graduation academic honors
                entry cards/invitations graduation rites
                graduate first semester or summer
                commence ceremony schedule
                Transcript of Records
                electronic Transcript of Records
                unofficial Transcript of Records
                requests Transcript of Records
                schedule Transcript of Records
                hold Transcript of Records
                diploma
                lost diploma
                Parent's
                data Registrar
                change grade
                failed deadline
                students not in class
                Where to pay
                When to pay?
                Story

                Text:

            """ + msg + "Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION."},
        ])
    print(msg)
    reply = response['choices'][0]['message']['content']
    print(reply)
    return reply