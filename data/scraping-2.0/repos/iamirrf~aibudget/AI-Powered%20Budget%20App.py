import openai
import pandas as pd
import os

# Initialize OpenAI API
openai.api_key = "YOUAPIKEY"

FILE_NAME = "users_data.xlsx"
current_user = None

# Utility functions
def save_to_excel(data):
    df = pd.DataFrame(data)
    df.to_excel(FILE_NAME)

def read_from_excel():
    if os.path.exists(FILE_NAME):
        return pd.read_excel(FILE_NAME, index_col=0).to_dict(orient='index')
    return {}

users = read_from_excel()

def register():
    global current_user
    username = input("Enter a username: ")
    if username in users:
        print("Username already exists.")
        return
    password = input("Enter a password: ")
    users[username] = {"password": password, "income": 0, "expenses": {}}
    current_user = username
    save_to_excel(users)
    print("Registration successful!")

def login():
    global current_user
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    if users.get(username) and users[username]["password"] == password:
        current_user = username
        print("Logged in successfully!")
    else:
        print("Invalid credentials!")

def add_income():
    income = float(input("Enter your monthly income: "))
    users[current_user]["income"] = income
    save_to_excel(users)

def add_expense():
    category = input("Enter expense category (e.g. Rent, Food, Entertainment): ")
    amount = float(input(f"Enter amount for {category}: "))
    users[current_user]["expenses"][category] = amount
    save_to_excel(users)

def view_budget():
    print(f"\nIncome: ${users[current_user]['income']}")
    print("Expenses:")
    for category, amount in users[current_user]['expenses'].items():
        print(f"{category}: ${amount}")
    total_expenses = sum(users[current_user]['expenses'].values())
    print(f"Total Expenses: ${total_expenses}")
    print(f"Remaining Budget: ${users[current_user]['income'] - total_expenses}\n")

def get_ai_suggestions():
    total_expenses = sum(users[current_user]['expenses'].values())
    remaining_budget = users[current_user]['income'] - total_expenses
    
    prompt = f"I have a monthly income of ${users[current_user]['income']} and expenses totaling ${total_expenses}. Here's a breakdown of my expenses: {users[current_user]['expenses']}. How can I achieve financial freedom?"

    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=150)
    print(response.choices[0].text.strip())

def menu():
    while True:
        print("\nAI-powered Budgeting App")
        print("1. Register")
        print("2. Login")
        print("3. Add Income")
        print("4. Add Expense")
        print("5. View Budget Overview")
        print("6. Get AI-Powered Financial Suggestions")
        print("7. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            register()
        elif choice == "2":
            login()
        elif choice == "3":
            if current_user:
                add_income()
            else:
                print("Please login first!")
        elif choice == "4":
            if current_user:
                add_expense()
            else:
                print("Please login first!")
        elif choice == "5":
            if current_user:
                view_budget()
            else:
                print("Please login first!")
        elif choice == "6":
            if current_user:
                get_ai_suggestions()
            else:
                print("Please login first!")
        elif choice == "7":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    menu()
