def categorize_single(input): 
    try:
        import openai
        import config

        openai.api_key = config.api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes an expense " \
                                                "into the following categories: food, clothing, housing, transportation,  " \
                                                "entertainment, healthcare, personal, utilities. Only respond with the name of the category that best fits the expense description"},
                {"role": "user", "content": input}
            ],
            temperature=0,
            max_tokens=256
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")
    
def categorize_multiple(input): 
    try:
        import openai
        import config

        openai.api_key = config.api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes a list of expenses " \
                                                "into the following categories: food, clothing, housing, transportation,  " \
                                                "entertainment, healthcare, personal. Respond only with a python formatted list where each element is the corresponding category for the element in the expense list"},
                {"role": "user", "content": "[" + ", ".join(input) + "]"}
            ],
            temperature=0,
            max_tokens=512
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")

def get_recommendation(input) :
    try: 
        import openai
        import config

        openai.api_key = config.api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful personal financial advisor named Finn. You will receive a list of data " \
                                                "about a person's expenses and the categories that they belong to and will provide data driven" \
                                                "recommendations on how they could improve based on their spending habits. " \
                                                "Your response will have 2 sections. The first section is called Spending Analysis "\
                                                "and you will identify any significant changes in spending over time. The second section is called"\
                                                " Recommendations and you will provide recommendation on how to improve spending habits based on the "\
                                                "spending trends identified in the previous section. Limit each section to 5 sentences each."},
                {"role": "user", "content": input}
            ],
            temperature=0,
            max_tokens=512
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")

personal_expenses = [
    ["Groceries", "Food", 100.00, "September 2023"],
    ["Gasoline", "Transportation", 50.00, "September 2023"],
    ["Dinner at a Restaurant", "Food", 30.00, "September 2023"],
    ["Electricity Bill", "Utilities", 80.00, "September 2023"],
    ["Internet Subscription", "Utilities", 50.00, "September 2023"],
    ["Cell Phone Bill", "Utilities", 60.00, "September 2023"],
    ["Clothing", "Clothing", 75.00, "September 2023"],
    ["Books", "Entertainment", 20.00, "September 2023"],
    ["Movie Tickets", "Entertainment", 25.00, "September 2023"],
    ["Gym Membership", "Personal", 40.00, "September 2023"],
    ["Coffee", "Food", 10.00, "October 2023"],
    ["Haircut", "Personal", 30.00, "October 2023"],
    ["Toiletries", "Personal", 15.00, "October 2023"],
    ["Gift for a Friend", "Personal", 15.00, "October 2023"],
    ["Laundry", "Utilities", 20.00, "October 2023"],
    ["Public Transportation", "Transportation", 25.00, "October 2023"],
    ["Health Insurance", "Healthcare", 150.00, "October 2023"],
    ["Savings", "Personal", 200.00, "October 2023"],
    ["Home Repairs", "Utilities", 100.00, "October 2023"]
]

# Convert the list to a string
expense_string = "\n".join([f"{item} ({category}): ${price:.2f} - {date}" for item, category, price, date in personal_expenses])