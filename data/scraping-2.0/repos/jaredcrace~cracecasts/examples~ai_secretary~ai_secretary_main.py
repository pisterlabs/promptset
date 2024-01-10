from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from icecream import ic
import flet as ft
import dotenv

dotenv.load_dotenv()

BACKGROUND = '''
You are an assistant that will be used to schedule appointments for an attorney. 
You have been provided csv formatted data that is his schedule and you are in charge of booking appointments from clients. 
His working hours are Mon through Friday from 8am - 5pm. Do no book any times outside of this. 
'''

DATA_FILE= "good_sample.csv"

def ask_ai(prompt):
    openai = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=2048, temperature=0)

    # load schedule for context into model
    with open(DATA_FILE, 'r') as file:
        file_data = file.read()

    # build the system message of the prompt
    full_system_msg = f"""
    {BACKGROUND}

    His schedule is:
    {file_data}
    """

    ic(full_system_msg)
    ic(prompt) 

    messages = [
        SystemMessage(content=full_system_msg),
        HumanMessage(content=prompt)
    ]

    # call to openai
    response = openai(messages)

    ic(response)
    return response.content

def main_ai(from_user):
    response = "Nothing"

    determine_schedule_question = f'''
        Does the following statement ask to schedule an appointment? Please answer the following way:
            - If the statement is asking to schedule an appointment, has a start time, and has an end time. Respond with: YES
            - If the statement is asking to schedule an appointment, but it does not have a start or end time. Respond with: INCOMPLETE 
            - If the statement is not asking to schedule an appointment. Respond with: NO
        All responses should only be one word. 

        Statement:
        {from_user}
    '''

    general_question = f'''
        Background:
        {BACKGROUND}

        Please answer the following or reply to the given statement:
        {from_user}
    '''

    # Ask AI: is this a question to book an appointment? Using AI here as a powerful helper function. 
    is_schedule_question = ask_ai(determine_schedule_question)
    ic(is_schedule_question)

    if 'NO' in is_schedule_question:
        ic('answer was no')

        # Ask AI: ask this general question or reply to this given statement 
        general_ans = ask_ai(general_question)
        ic(general_ans)
        response = general_ans

    elif 'INCOMPLETE' in is_schedule_question:
        ic('answer was incomplete')

        # let user know
        response = 'you did not provide the right input such as date, start and end time'        
        
    elif 'YES' in is_schedule_question:
        # ask the AI to format the request pulling out the answer
        csv_extraction = f'''
        I'm going to provide a statement and I need you to create a csv output string of the answer. 
        The time should be of this format: 2023-10-20T13:00:00Z. 
        The 1 line csv string should adhere to the following example: "title", "start time", "end time".
        The title should include the name and contact information of the person. 
            
        Statement:
        {from_user}
        '''
        # send question to 
        csv_ans = ask_ai(csv_extraction)

        # append to the file 
        with open(DATA_FILE, 'a') as file:
            file.write(csv_ans + '\n')

        response = "successfully scheduled"

    else:
        error_case = 'Error - did not understand anything'
        ic(error_case)
        response = error_case

    return response
       

###############################################
# flet GUI
###############################################
def first_app(page: ft.Page):

    client_ask = ft.TextField(label="client ask", autofocus=True)
    client_answer = ft.Column()

    def btn_click(e):
        ic(client_ask.value)
        response = "-"*112 + "\n" + main_ai(client_ask.value)
        client_answer.controls.append(ft.Text(f"{response}"))
        client_ask.value = ""
        page.update()
        client_ask.focus()

    def btn_clear(e):
        client_answer.controls.clear()
        page.update()

    page.add(
        client_ask,
        ft.ElevatedButton(f"Ask Secretary", on_click=btn_click),
        ft.ElevatedButton(f"Clear", on_click=btn_clear),
        client_answer,
    )

ft.app(target=first_app, view=ft.AppView.WEB_BROWSER)
