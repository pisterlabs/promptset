import ast
import os
import openai
import logging
import json
import datetime

from sms_app.models import User

LOGGER = logging.getLogger('friday_logger')

openai.apikey = os.getenv("OPENAI_API_KEY")

################
# Top Classifier
################
def determine_conversation_category(text):
    """
    Takes an input text and categorizes the intent of the query.
    Classes: Chat, Info, Setup, Transaction, & Expense inquiry.
    """
    response = openai.Classification.create(
        search_model="davinci",
        model="davinci",
        query=text,
        examples=[
            # More Info needed examples
            #["I'd like to set up my budget", User.ASK],
            #["I want to setup my budget", User.ASK],
            #["Can I setup my budget?", User.ASK],
            #["Can I set up my budget?", User.ASK],
            #["Track a transaction", User.ASK],
            #["Budget for me", User.ASK],
            #["Make a budget", User.ASK],
            #["I'm going to record a transaction", User.ASK],
            #["Track some spending for me", User.ASK],
            # Setup examples
            ["Add a budget of 100 dollars for Transportation items", User.SET],
            ["Make a new category for me called Health", User.SET],
            ["Change my budget category Health to allow me to spend 56 dollars", User.SET],
            # Transaction examples
            ["I spent 10 dollars at McDonalds today", User.TRA],
            ["I bought a brand new TV at BestBuy yesterday", User.TRA],
            ["I got some sneakers for a friend yesterday, it cost about $200", User.TRA],
            # Expense inquiry examples
            ["What is my spending breakdown?", User.INQ],
            ["What are my budget categories?", User.INQ],
            ["How much have I spent for the past 30 days", User.INQ],
            ["How much money do I have remaining in my total budget", User.INQ],
            # Discussion
            ["Tell me about yourself", User.DIS],
            ["What does this app do?", User.DIS],
            ["What are you?", User.DIS],
            ["I love you so much", User.DIS],
            ["How are you doing?", User.DIS],
            ["I need someone to talk to", User.DIS],
            ["Hey so I need some inspiration to start my day", User.DIS],
            ["Do you have any advice for me to improve my spending habits?", User.DIS]
        ],
        labels=[User.DIS, User.SET, User.TRA, User.INQ]
    )
    
    return response["label"]


#################
# About functions
#################
def get_about_response(question, user): 
    """
    Takes a user input question and the user's name
    and generates a response about what friday can do
    """
    # Start here:
    prompt = \
"""
The following is a question and answer dicussion with an AI assistant named Friday and the user %s.
The AI must be able to inform %s of all the functionality that Friday can perform.
The AI assistant has the functionality to help %s with these actions:
- track spending
- allocate budgets
- setup categories for budgeting
- visualize spending with graphs
- Give reports on spending
- have helpful discussions about budgeting and finance
- provide inspiration

""" % (user, user, user)

    prompt += '%s: %s\n' % (user, question)
    prompt += 'Friday:'

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.6,
        max_tokens=132,
        top_p=1,
        frequency_penalty=1.7,
        presence_penalty=1.7,
        stop="%s:" % user
    )
    return response.choices[0].text.strip()


######################
# Discussion functions
######################
def get_discussion_response(conv_history, user):
    """
    Takes a list of dictionaries containing the Author and Msg
    in order and generates a response
    """

    # Create a String prompt
    prompt = \
"""
The following is a conversation with an AI assistant named Friday and the user %s.
The AI assistant is helpful, clever, optimistic, motivational, reflective and very friendly.
The AI assistant will mostly focus on topics related to budgeting, personal finance, saving and earning money.
The purpose of the AI assistant should inspire %s to save money and spend money wisely.
The AI assistant should provide insightful information and new ideas to meet %s's budgeting goals and new ways to earn money.
Friday does not spend any money herself. Friday's current timezone is Eastern Standard time.
Friday should answer in a short text about a sentence long.
Friday must be able to inform %s of all the functionality that she can perform.
The AI assistant has the functionality to help %s with these actions:
- track spending
- allocate budgets
- setup categories for budgeting
- Give reports on spending
- have helpful discussions about budgeting and finance
- provide inspiration

""" % (user, user, user, user, user)

    # Populate with conversation
    for message in conv_history:
        prompt += '%s:%s\n' % (message['Author'], message['Message'])

    prompt += 'Friday:'

    # Send to API
    response_dict = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=1.0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=2.0,
        presence_penalty=2.0,
        stop="%s:" % user
    )

    # Clip any enters off the ends of the string
    response = response_dict['choices'][0]['text'].strip()

    return response


######################
# User setup functions
######################
def determine_user_setup_intent(text):
    """
    Determines what type of setup the user would like to do
    """
    response = openai.Classification.create(
        search_model="davinci",
        model="davinci",
        query=text,
        examples=[
            # Info examples
            ["Delete the Housing category", "Change budget"],
            ["I want to setup my budget", "Change budget"],
            ["Can I setup my budget?", "Change budget"],
            # Unknown examples
            ["Change my phone number", "Unclear"],
            ["Make my budget bigger", "Unclear"],
            ["Change my name to something cool", "Unclear"],
        ],
        labels=["Change name", "Change budget", "Unclear"]
    )
    
    return response["label"]

def get_budget_response(msg, budget_dict, user):
    """
    Determines how to edit the users budget categories based on a msg
    """
    budget_dict = str(budget_dict).replace("'", '"')
    prompt = \
"""
Below is a program that modify the data of %s's budgeting categories based on his request.
Please modify the dictionary according to the request.
Each request is independent.
If you can't find the budget, don't change anything.
Set newly created budgets to a value of 'N/A'
Also write a happy message to %s informing him of the change in a sentence.

Categories: {"Personal Care": "N/A", "Clothing": "N/A", "Gifts": "N/A", "Education": "N/A", "Holiday": "200", "Donation": "200", "Emergencies: "200", "Medical": "100"}
Prompt: Add a new section called Restaurants and allocate 500 dollars to it
Output: {"Categories" : {"Personal Care": "N/A", "Clothing": "N/A", "Gifts": "N/A", "Education": "N/A", "Holiday": "200", "Donation": "200", "Emergencies: "200", "Medical": "100", "Restaurants": "500"},
 "Response" : "You've added a new section called Restaurants and allocated 500 dollars to it!"}

Categories: {"Personal Care": "N/A", "Clothing": "N/A", "Gifts": "N/A", "Housing" : "N/A", "Transportation": "100"}
Prompt: Delete my budget for transportation
Output: {"Categories" : {"Personal Care": "N/A", "Clothing": "N/A", "Gifts": "N/A", "Housing": "N/A"},
"Response": "Your transportation budget has been deleted!"}

Categories: {"Personal Care": "N/A", "Clothing": "N/A", "Gifts": "N/A", "Housing" : "N/A", "Transportation": "100", "Housing" : "N/A", "Food": "24.00"}
Prompt: Change my healthcare budget
Output: {"Categories": {"Personal Care": "N/A", "Clothing": "N/A", "Gifts": "N/A, ""Housing" : "N/A", "Transportation": "100", "Housing" : "N/A", "Food": "24.00"},
"Response": "I couldn't find a budget called healthcare."}

Categories: { "Education": "N/A", "Holiday": "200", "Donation": "200", "Emergencies: "200", "Medical": "100", "Restaurants": "500"}
Prompt: Create a transportation budget
Output: {"Categories" : { "Education": "N/A", "Holiday": "200", "Donation": "200", "Emergencies: "200", "Medical": "100", "Restaurants": "500", "Transportation": "N/A"},
"Response": "You've created a transportation budget!"}

-------------------------------------------

Categories: %s
Prompt: %s
Output:
""" % (user, user, budget_dict, msg)

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=296,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop='Output:'
    )

    data = json.loads(response.choices[0].text.strip())
    return data


######################
# Transaction functions
######################
def determine_transaction_info(text):
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    three_days_ago = today - datetime.timedelta(days=3)
    a_week_ago = today - datetime.timedelta(days=7)
    prompt = \
        """A classifier that can indicate the item, location, date, and amount for a transaction in a list format. If you 
    don't know what the value for a category is, simple use '?' as it's value. Program should return a list 
    containing a dictionary for each transaction. The program should be able to determine the date relative to today from a sentence.
    today is %s-%s-%s

Sentence: Hey Joe, I bought some band-aids at shoppers today for 10 bucks. I also bought 50 k worth of chicken wings 
at Kentucky fried chicken. 
List: [{'Item': 'Band-aids', 'Amount': 10, 'Location': 'Shoppers', 'Date' : '%s-%s-%s'}, {'Item': 'Chicken Wings', 'Amount': 50000, 'Location': 'Kentucky fried 
chicken', 'Date' : '%s-%s-%s'}] 

Sentence: Yo, I spend 40 dollars at McDonalds today on a Big Mac then I went to no frills for some bananas, 
it costed 10 bucks 
List: [{'Item': 'Big Mac', 'Amount': 40, 'Location': 'McDonalds', 'Date' : '%s-%s-%s'}, {'Item': 'Bananas', 'Amount': 10, 'Location': 'No frills', 'Date' : '%s-%s-%s'}]

Sentence: yesterday I bought a new pair of shoes at shoppers for 20 bucks. 
List: [{'Item': 'Shoes', 'Amount': 20, 'Location': 'Shoppers', 'Date' : '%s-%s-%s'}]

Sentence: Dude, I dropped 20 k on a new iPad today it was so expensive and bro I got shopping today and got some groceries at Sobeys for like 30 bucks oh and the iPad was from Apple yeah and uh yesterday I bought this new app on steam it was like this indie game and it was really fun to play its called apex legends or something but this one was no biggie only like 23 or something 
List: [{'Item': 'iPad', 'Amount': 20000, 'Location': 'Apple', 'Date' : '%s-%s-%s'}, {'Item': 'Groceries', 'Amount': 30, 'Location': 'Sobeys', 'Date' : '%s-%s-%s'}, {'Item': 'Apex Legends', 'Amount': 23, 'Location': 'Steam', 'Date' : '%s-%s-%s'}] 

Sentence: I bought a burger at McDonalds 3 days ago
List: [{'Item': 'Burger', 'Amount': '?', 'Location': 'McDonalds', 'Date' : '%s-%s-%s'}]

Sentence: I bought a shirt for $15 at Walmart a week ago.
List: [{'Item': 'Shirt', 'Amount': 15, 'Location': 'Walmart', 'Date' : '%s-%s-%s'}]

Sentence: %s
List:
    """ % (today.month, today.day, today.year,
           today.month, today.day, today.year,
           today.month, today.day, today.year,
           today.month, today.day, today.year,
           today.month, today.day, today.year,
           yesterday.month, yesterday.day, yesterday.year,
           today.month, today.day, today.year,
           today.month, today.day, today.year,
           yesterday.month, yesterday.day, yesterday.year,
           three_days_ago.month, three_days_ago.day, three_days_ago.year,
           a_week_ago.month, a_week_ago.day, a_week_ago.year,
           text)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop='Sentence:'
    )
    try:
        print(response.choices[0].text.strip())
        transactionList = ast.literal_eval(response.choices[0].text.strip())
        return transactionList
    except:
        return None

def determine_transaction_type(text, categories):
    """
    Categorizes reason for transaction
    """
    response = openai.Classification.create(
        search_model="davinci",
        model="davinci",
        query=text,
        examples=[
            # Housing
            ["house", "Housing"],
            ["electric bill", "Housing"],
            ["utility bill", "Housing"],

            # Household Supplies
            ["couch", "Household"],
            ["shelf", "Houshold"],
            ["curtain rod", "Household"],

                # Clothing
            ["dress", "Clothing"],
            ["shoes", "Clothing"],
            ["rain jacket", "Clothing"],

                # Education
            ["textbook", "Education"],
            ["tuition", "Education"],
            ["iclicker", "Education"],

                # Groceries
            ["milk", "Groceries"],
            ["water", "Groceries"],
            ["doritos", "Groceries"],

            # Transportation
            ["car", "Transportation"],
            ["bus", "Transportation"],
            ["drive", "Transportation"],

            # Personal Care
            ["gel", "Health"],
            ["shampoo", "Health"],
            ["deoderant", "Health"],
            ["gym equipment", "Health"],
            ["Bike", "Health"],

        ],
        labels=categories
    )

    return response["label"]

#####################
# Elaborate functions
#####################
def get_elaboration_response(msg, user):
    """
    Prompts user to elaborate more about what they would like to do
    """
    # Start here:
    prompt = \
"""
Assume the sentence is incomplete, and the task is to ask %s to elaborate with more completely with information about what he or she wants to do.

%s: I would like to delete the category
AI: I can do that for you! Can you tell me a little more about what you would like me to do?
""" % (user, user)

    prompt += '%s: %s\n' % (user, msg)
    prompt += 'AI:'
    LOGGER.info(prompt)

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.6,
        max_tokens=132,
        top_p=1,
        frequency_penalty=1.7,
        presence_penalty=1.7,
        stop="%s:" % user
    )
    return response.choices[0].text.strip()

######################
# Inquiry functions
######################
def determine_inquiry_type(msg):
    """
    Categorizes whether the inquiry is looking for a visual or text response
    """
    response = openai.Classification.create(
        search_model="davinci",
        model="davinci",
        query=msg,
        examples=[
            # Visual
            ["Can I see some visuals regarding my expenses", "Visual"],
            ["I'd like to see a pie chart of my budget", "Visual"],
            ["I'd like to see a scatter plot of my spendings for the past month", "Visual"],
            ["I'd like to see a plot of my expenses", "Visual"],
            ["I'd like to see a graph of my spendings", "Visual"]

            # Text
            ["How much have spent for the past week", "Text"],
            ["How much did I spend yesterday", "Text"],
            ["Let me see an expenditure log of my recent transactions", "Text"],

        ],
        labels= [" Visual", "Text"]
    )

    return response["label"]

def get_visuals():
    pie_chart = "visualizations.piechart_visualization(example_dict)"
    pass

def get_inquiry_response_alt(user, msg, budget_list, total_spent, total_left, total_budget, trans_hist, overall_status):
    """
    Gets a response about the spending of the user when prompted
    """
    if total_budget == 0:
        total_left = 'No Budget Set'

    budgets = \
"""
Budgets for this month
------------------------
"""

    for budget_cat in budget_list:
        for name, value in budget_cat.items():
            budgets += '%s : %s\n' % (name, value)
        budgets += '\n'

    budgets += \
"""

Totals this month
------------------------
"""
    budgets += 'Money from budget left to spend: %s\nTotal Money Spent: %s\nStatus: %s' % (total_left, total_spent, overall_status)
    
    transactions = \
"""

Transactions this month
------------------------
"""
    if trans_hist is None:
        transactions += 'There is no transaction history'

    for trans in trans_hist:
        for name, value in trans.items():
            if name != 'DateTime':
                transactions += '%s : %s\n' % (name, value)
        transactions += '\n'
    
    transactions += \
"""

Totals this month
------------------------
"""
    transactions += 'Money from budget left to spend: %s\nTotal Money Spent: %s\nStatus: %s' % (total_left, total_spent, overall_status)
    
    response = openai.Classification.create(
        search_model="davinci",
        model="davinci",
        query=msg,
        examples=[
            # Text
            ["Show me my spending", "Transactions"],
            ["How much did I spend yesterday?", "Transactions"],
            ["Let me see an expenditure log of my recent transactions", "Transactions"],
            ["What are my budgets this week?", "Budgets"],
            ["Show me my budget categories", "Budgets"],
            ["How much do I have left in my budget?", "Budgets"]

        ],
        labels= ["Budgets", "Transactions"]
    )

    if response['label'] == 'Budgets':
        return budgets
    else:
        return transactions

def get_inquiry_response(user, msg, budget_list, total_spent, total_left, total_budget, trans_hist, overall_status):
    """
    Gets a response about the spending of the user when prompted
    """
    date = datetime.datetime.today()
    prompt = \
"""
The following is a conversation with an AI assistant named Friday and the user %s.
The AI assistant is helpful, clever, optimistic, motivational, reflective and very friendly.
The table below contains the information all of %s's spending history and budget information.
Use this table to create a detailed, human-like, and enthusiastic response to describe the contents of this table according to Mark's request.
The date is in the format of month/day/year

Date today: %s/%s/%s
""" % (user, user, date.month, date.day, date.year)

    if total_budget == 0:
        total_left = 'No Budget Set'

    prompt += \
"""
Budgets for this month
------------------------
"""

    for budget_cat in budget_list:
        for name, value in budget_cat.items():
            prompt += '| %s : %s' % (name, value)
        prompt += '\n'
    
    prompt += \
"""

Transactions this month
------------------------
"""
    if trans_hist is None:
        prompt += 'There is no transaction history'

    for trans in trans_hist:
        for name, value in trans.items():
            if name != 'DateTime':
                prompt += '| %s : %s' % (name, value)
        prompt += '\n'
    
    prompt += \
"""

Totals this month
------------------------
"""
    prompt += 'Money from budget left to spend: %s | Total Money Spent: %s | Status: %s' % (total_left, total_spent, overall_status)
    
    prompt += \
"""

Conversation
------------------------
"""
    prompt += '\n%s : %s\n' % (user, msg)
    prompt += 'Friday :'

    LOGGER.info(prompt)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["%s" % user]
    )

    return response.choices[0].text.strip() 

########################
# Other useful funcitons
########################
def find_name_from_msg(msg):
    """
    Decifer message to find name, if not found return None
    """
    prompt = \
"""
Only return the name from the sentence below, if name not found return None
Also, if the name is Friday, return None

Sentence: I'm Bob
Name: Bob
Sentence: T'is I, Helen Mortimer
Name: Helen
Sentence: Hey, its travis speaking
Name: Travis
Sentence: How are you doing?
Name: None
Sentence: My name is Friday
Name: None
Sentence: bill is my name
Name: Bill
Sentence: Will is too good
Name: None
Sentence: Dollar
Name: Dollar
Sentence: riddy
Name: Riddy
Sentence: hello
Name: Hello
Sentence: %s
Name:
""" % msg

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop="Sentence:"
    )
    name = response.choices[0].text.strip()
    if name == 'None':
        return None
    return name

    
