from langchain.llms import GooglePalm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from call_data_apis import create_user_api, remove_update_category_api
from utils import find_email, check_email, get_availaible_categories
import os

import re
os.environ['GOOGLE_API_KEY'] =  'AIzaSyAJuIFT_1XfowTRQH_qP5VC9ip8VTdyNKs'




def ask_anything():
    flag2 = False
    while True:
        response1 = "Bot: is there anything else i can help you with?\nPlease only answer in 'yes' or 'no'"
        print(response1+"\n"+"-"*10)
        user_answer = input("You: ")
        if user_answer.strip().lower() == 'yes':
            response1 = "Bot: what can i help you with?"
            print(response1+"\n"+"-"*10)
            flag2 = True
            break
        elif user_answer.strip().lower() == 'no':
            response1 = "Bot: Thank you for using statudos"
            print(response1+"\n"+"-"*10)
            break
        else:
            response1 = "Bot: Sorry I dont understand"
            print(response1+"\n"+"-"*10)
            continue
    return flag2

def Create_Account(categories_names,categories):
    record_reponse_create_account ={"name":None,"email":None,"categories":None}
    force_exit = False
    while record_reponse_create_account["name"]==None or record_reponse_create_account["email"]==None or record_reponse_create_account["categories"]==None:
        flag = False
        questions = ["What is your name?","what is your email?","what is your categories?"]
        while True:
            for pos,q in enumerate(questions):
                if  record_reponse_create_account["name"] != None and questions[pos].split(" ")[-1][:-1] =="name":
                    continue
                if record_reponse_create_account["email"] != None and questions[pos].split(" ")[-1][:-1] =="email":
                    continue
                if record_reponse_create_account["categories"] != None and questions[pos].split(" ")[-1][:-1] =="categories":
                    continue
                response1 = f"Bot: {q}"
                print(response1+"\n"+"-"*10)
                user_answer = input("You: ")
                if user_answer.strip().lower() == 'exit':
                    force_exit = True
                    break
                if questions[pos].split(" ")[-1][:-1] =="name":
                    record_reponse_create_account["name"] = user_answer
                elif questions[pos].split(" ")[-1][:-1] =="email":
                    record_reponse_create_account["email"] = user_answer
                elif questions[pos].split(" ")[-1][:-1] =="categories":
                    record_reponse_create_account["categories"] = user_answer
            
            while True:
                if (record_reponse_create_account["name"] != None and record_reponse_create_account["email"] != None and record_reponse_create_account["categories"] != None):
                    response1 = f"Bot: Is the Given this info is right?\nName: {record_reponse_create_account['name']}\nEmail: {record_reponse_create_account['email']}\nCategories: {record_reponse_create_account['categories']}\nPlease only answer in 'yes' or 'no'\if you want to exit type 'exit'"
                    print(response1+"\n"+"-"*10)
                    user_answer = input("You: ")
                    if user_answer.strip().lower() == 'yes':
                        flag = True
                        break
                    elif user_answer.strip().lower() == 'no':
                        my_flag = False
                        response1 = "Bot: what is wrong?\nPlease only answer in 'name' or 'email' or 'categories' or 'exit'\ne.g name\nor\nname,email\nor\nname,email,categories\nor\nexit"
                        print(response1+"\n"+"-"*10)
                        while True:
                            user_answer = input("You: ")
                            if user_answer.strip().lower() == 'exit':
                                force_exit = True
                                flag = True
                                my_flag = True
                                break
                            elif "name" in user_answer.strip().lower() and "email" in user_answer.strip().lower() and "categories" in user_answer.strip().lower():
                                record_reponse_create_account["name"] = None
                                record_reponse_create_account["email"] = None
                                record_reponse_create_account["categories"] = None
                                my_flag = True
                                break
                            elif "name" in user_answer.strip().lower() and "email" in user_answer.strip().lower() :
                                record_reponse_create_account["name"] = None
                                record_reponse_create_account["email"] = None
                                my_flag = True
                                break
                            elif "name" in user_answer.strip().lower() and "categories" in user_answer.strip().lower() :
                                record_reponse_create_account["name"] = None
                                record_reponse_create_account["categories"] = None
                                my_flag = True
                                break
                            elif "email" in user_answer.strip().lower() and "categories" in user_answer.strip().lower() :
                                record_reponse_create_account["email"] = None
                                record_reponse_create_account["categories"] = None
                                my_flag = True
                                break
                            elif user_answer.strip().lower() == 'name':

                                record_reponse_create_account["name"] = None
                                my_flag = True
                                break
                            elif user_answer.strip().lower() == 'email':
                                record_reponse_create_account["email"] = None
                                my_flag = True
                                break
                            elif user_answer.strip().lower() == 'categories':
                                record_reponse_create_account["categories"] = None
                                my_flag = True
                                break
                            else:
                                response1 = "Bot: Sorry I dont understand"
                                print(response1+"\n"+"-"*10)
                                continue
                        if my_flag:
                            break
                    elif user_answer.strip().lower() == 'exit':
                        force_exit = True
                        flag = True
                        break
                    else:
                        response1 = "Bot: Sorry I dont understand"
                        print(response1+"\n"+"-"*10)   
                        continue
            if flag:
                break
        

        
    if force_exit==False and (record_reponse_create_account["name"] != None and record_reponse_create_account["email"] != None and record_reponse_create_account["categories"] != None):
        # print(record_reponse_create_account)
        user_account_cats = []
        not_avail_categories = []
        for user_cat in record_reponse_create_account["categories"].split(","):
            if user_cat.lower().strip() in categories.keys():
                user_account_cats.append(categories[user_cat.lower().strip()])
            else:
                not_avail_categories.append(user_cat.lower().strip())
        if len(not_avail_categories)>0:
            response1 = f"Bot: Sorry these categories are not available {','.join(not_avail_categories)}\nAvailable categories are {categories_names}"
            print(response1+"\n"+"-"*10)
        if len(user_account_cats)>0:
            input_data = {'userName': record_reponse_create_account["name"], 'email': record_reponse_create_account["email"], 'categories': user_account_cats}
            output_data = create_user_api(input_data)
            # print(output_data)
            if output_data is not None:
                if output_data["message"] != "user with this email already exists":
                    response1 = "Bot: Thank you for creating an account"
                    print(response1+"\n"+"-"*10)
                    response1  ="Bot: Your account has been created successfully"
                    print(response1+"\n"+"-"*10)
                elif output_data["message"] == "user with this email already exists":
                    response1 = "Bot: User with this email is already exists"
                    print(response1+"\n"+"-"*10)
                else:
                    response1 = "Bot: Failed to create an account"
                    print(response1+"\n"+"-"*10)
            else:
                    response1 = "Bot: Failed to create an account"
                    print(response1+"\n"+"-"*10)
        
        flag2  = ask_anything()
        return flag2,record_reponse_create_account
    if flag2:
        flag2=ask_anything()
        return flag2,record_reponse_create_account
    return False,record_reponse_create_account
def get_conversational_chain(vector_store):
    llm = GooglePalm()
    
    # Set up a system message to start the conversation
    system_message = "Hi I am statudos Bot, How can i help you today?"
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    
    # Start the conversation with the system message
    conversation_chain({'question': system_message})
    
    return conversation_chain



    

def chat_with_palm2():
    categories_names,categories = get_availaible_categories()
    print("Type 'exit' to end the conversation.")
#     instruct = f"""Statudos is a website where poeple get status of influencers based on their personal categories({categories_names}) and you are a statudos wastapp bot that provide info regarding statudo's services and do greetings with users and helps user to see availaible categories,user personal categories, create account, provide information and update or change their categories.
#     'if user wants to create an account(return 'create_account") and if user wants to check availaible categories  (return 'check_categories') and if user wants to change or update his/her own categories(return 'update_category') and if user want to check his/her own categories(return 'my_categories') and people will use to comment on status(if comment in user query so return 'comment').
#    ,Once you return these terms(create_account,check_categories,update_categories,my_categories) based on user's reply you have recognized user intent in these terminologies you dont have to do these terms actions by yourself just return these terms based on user's intents'.if you dont know anything dont try to create answer by yourself and dont answer anything that other is than statudos.
#     """
    instruct = f"""Statudos is a website providing information about influencers and their status based on personal categories ({categories_names}). You are a Statudos WhatsApp bot with the role of offering details about Statudos' services, greeting users, and assisting them in various tasks. These tasks include checking available categories, viewing personal categories, creating an account, retrieving information, and updating or changing categories.

If a user wants to create an account, respond with 'create_account'.If a user wishes to update or change categories, respond with 'update_category'.To check their own categories, respond with 'my_categories' Users can also comment on status updates.To remove their own categories, respond with 'remove_categories', and if the user's query contains a comment, respond with 'comment.'

Your role is to talk with politely and do greetings with user and tell user about your services if user ask and recognize user intent and return one of these terms ('create_account',  'update_category', 'my_categories', 'comment') based on the user's input and Additionally, provide information to users regarding Statudos . If a user inquires about topics other than Statudos or its services or terms, simply respond with 'I don't know.' Do not perform the actions associated with the terms; only identify and return the relevant term based on the user's intentions.

If you are unsure or lack information about a particular query, refrain from generating answers beyond the context of Statudos.
"""


    vector_store = FAISS.from_texts([instruct], embedding=GooglePalmEmbeddings())
    conversation_chain = get_conversational_chain(vector_store)
    already_answr = False
    user_email = None
    while True:
        if not already_answr:
            user_answer = input("You: ")

        
        if user_answer.lower() == 'exit':
            response1 = "Bot: Goodbye! Have a great day."
            print(response1+"\n"+"-"*10)
            break
        already_answr = False
        # Process user input and get the bot's response
        response = conversation_chain({'question': user_answer})
        chat_history = response['chat_history']

        # print("CHat Historrrrrrrrrrrrrrrry")
        # print(chat_history)
        if chat_history[-1].content.lower().strip() == 'create_account':
            print("********************************Create Account*********************************")

            print(chat_history[-1].content.lower().strip(),"Entered in Manual Bot")
            flag,record_reponse_create_account = Create_Account(categories_names=categories_names,categories=categories)
            # print("**********",record_reponse_create_account)
            if flag:
                continue
            else:
                response1 = "Bot: Thank you for using statudos"
                print(response1+"\n"+"-"*10)
                break
        elif chat_history[-1].content.lower().strip() == 'update_category':
            print("********************************Update Categories*********************************")
            if user_email==None:
                while True:
                    response1 = "Bot: To update your categories please provide your email?"
                    print(response1+"\n"+"-"*10)
                    user_answer = input("You: ")
                    email = None
                    r = find_email(user_answer)
                    if r!=None:
                        response1 = f"Bot: is this email is correct or not?\n{r}\nPlease only answer in 'yes' or 'no'"
                        print(response1+"\n"+"-"*10)
                        user_answer = input("You: ")
                        if user_answer.strip().lower() == 'yes':
                            user_email = r
                            email = r
                            break
                        elif user_answer.strip().lower() == 'no':
                            continue
                        else:
                            already_answr = True
                            break
                    
            else:
                email = user_email
            if email!= None:
                res = check_email(email)
                if res[0]==True:
                    avail_category= list(res[1].keys())
                    
                    response1 = f"Bot: your categories are {','.join(avail_category)}\nDo you want to update your categories?\nplease only answer in 'yes' or 'no'"
                    print(response1+"\n"+"-"*10)

                    user_answer = input("You: ")
                    if user_answer=="yes":
                        response1 = f"Bot: Please provide your new categories?\nAvailable categories are {categories_names}"
                        print(response1+"\n"+"-"*10)
                        user_answer = input("You: ")
                        user_account_cats = []
                        for cat in categories_names.split(","):
                            if cat.lower().strip() in user_answer.lower().strip():
                                user_account_cats.append(categories[cat.lower().strip()])
                        if len(user_account_cats)>0:    
                            update_res = remove_update_category_api({'email':email,'categories':user_account_cats})
                            if update_res.get("message")!="categories with this id not found":
                                response1 = "Bot: Your Account Categories are updated"
                                print(response1+"\n"+"-"*10)
                                if update_res.get("data").get("nonExistingCategories"):
                                    response1 = f"Bot: These categories are not available: {','.join(update_res.get('data').get('nonExistingCategories'))}\nAvailable categories are: {categories_names}"
                                    print(response1+"\n"+"-"*10)
                            elif update_res.get("message")=="categories with this id not found":
                                response1 = f"Bot: {update_res.get('message')}"
                                print(response1+"\n"+"-"*10)
                            else:
                                response1 = "Bot: Failed to update your categories"
                                print(response1+"\n"+"-"*10)
                        else:
                            already_answr = True
                        print("Your Account Categories are updated")
                    elif user_answer=="no":
                        response1 = "Bot: So how may i help you now?"
                        print(response1+"\n"+"-"*10)
                    elif user_answer.strip().lower() == 'exit':
                        response1 = "Bot: Goodbye! Have a great day."
                        print(response1+"\n"+"-"*10)
                        break
                    else:
                        already_answr = True
                else:
                    response1 = f"Bot: Sorry there is no Account with this email: {email}\n if you want to create an account type create_account"
                    print(response1+"\n"+"-"*10)
            else:
                already_answr = True
        elif chat_history[-1].content.lower().strip() == 'my_categories':
            print("********************************My Categories*********************************")
            if user_email==None:
                while True:
                    response1 = "Bot: To update your categories please provide your email?"
                    print(response1+"\n"+"-"*10)
                    user_answer = input("You: ")
                    email = None
                    r = find_email(user_answer)
                    if r!=None:
                        response1 = f"Bot: is this email is correct or not?\n{r}\nPlease only answer in 'yes' or 'no'"
                        print(response1+"\n"+"-"*10)
                        user_answer = input("You: ")
                        if user_answer.strip().lower() == 'yes':
                            user_email = r
                            email = r
                            break
                        elif user_answer.strip().lower() == 'no':
                            continue
                        else:
                            already_answr = True
                            break
            else:
                email = user_email

            if email!= None:
                res = check_email(email)
                if res[0]==True:
                    avail_category= list(res[1].keys())
                   
                    print(avail_category)
                    response1 = f"Bot: your categories are {','.join(avail_category)}"
                    print(response1+"\n"+"-"*10)

                else:
                    response1 = f"Bot: Sorry there is no Account with this email: {email}\n if you want to create an account type 'I want to create_account'"
                    print(response1+"\n"+"-"*10)
            else:
                already_answr = True
        elif chat_history[-1].content.lower().strip() == 'remove_categories':
            print("********************************Remove Categories*********************************")
            if user_email==None:
                while True:
                    response1 = "Bot: To update your categories please provide your email?"
                    print(response1+"\n"+"-"*10)
                    user_answer = input("You: ")
                    email = None
                    r = find_email(user_answer)
                    if r!=None:
                        response1 = f"Bot: is this email is correct or not?\n{r}\nPlease only answer in 'yes' or 'no'"
                        print(response1+"\n"+"-"*10)
                        user_answer = input("You: ")
                        if user_answer.strip().lower() == 'yes':
                            user_email = r
                            email = r
                            break
                        elif user_answer.strip().lower() == 'no':
                            continue
                        else:
                            already_answr = True
                            break
            else:
                email = user_email

            if email!= None:
                res = check_email(email)
                if res[0]==True:
                    avail_category= list(res[1].keys())
                    response1 = f"Bot: your categories are {','.join(avail_category)}\nwhich categories you want to remove?"
                    print(response1+"\n"+"-"*10)
                    user_answer = input("You: ")
                    user_account_cats = []
                    for cat in user_answer.split(","):
                        if cat.lower().strip() in avail_category:
                            user_account_cats.append(categories[cat.lower().strip()])
                    if len(user_account_cats)>0:
                        remove_res = remove_update_category_api({'email':email,'categories':user_account_cats},remove=True)
                        if remove_res.get("message")!="categories with this id not found":
                            response1 = "Bot: Your Account Categories are removed"
                            print(response1+"\n"+"-"*10)
                            if remove_res.get("data").get("nonExistingCategories"):
                                response1 = f"Bot: These categories are not available: {','.join(remove_res.get('data').get('nonExistingCategories'))}\nAvailable categories are: {categories_names}"
                                print(response1+"\n"+"-"*10)
                        elif remove_res.get("message")=="categories with this id not found":
                            response1 = f"Bot: {remove_res.get('message')}"
                            print(response1+"\n"+"-"*10)
                        else:
                            response1 = "Bot: Failed to remove your categories"
                            print(response1+"\n"+"-"*10)
                else:
                    response1 = f"Bot: Sorry there is no Account with this email: {email}\n if you want to create an account type 'I want to create_account'"
                    print(response1+"\n"+"-"*10)
            
        else:
            print("Bot:", chat_history[-1].content+"\n"+"-"*10)
        

if __name__ == "__main__":
    chat_with_palm2()
