from flask import Blueprint, request, jsonify
from datetime import datetime
import json
from datetime import timedelta
from dateutil.relativedelta import relativedelta  # Use dateutil for date operations
import os
import openai


from services.mongo import MongoService

prod_bp = Blueprint("prod", __name__)
mongo = MongoService()
mongo.connect()

with open("output.json", "r") as file:
    data = json.load(file)

transaction_data = data["data"]

for transaction in transaction_data:
    transaction["date"] = datetime.strptime(
        transaction["date"], "%a, %d %b %Y %H:%M:%S %Z"
    )


# Helper function to filter transactions within a date range
def filter_transactions(transactions, start_date, end_date):
    return [
        transaction
        for transaction in transactions
        if start_date <= transaction["date"] <= end_date
    ]


# Edits transaction amount
@prod_bp.route("/api/prod/transaction/<transaction_id>", methods=["PUT"])
def update_transaction_amount(transaction_id):
    new_amount = request.json.get("modifiedAmount")

    if new_amount is None:
        return (
            jsonify(
                {"error": "The 'modifiedAmount' field is missing from the request."}
            ),
            400,
        )

    # Use the transaction_id to find the transaction in the database
    transaction = mongo.find_transaction_by_id(transaction_id)

    if transaction is None:
        return jsonify({"error": "Transaction not found."}), 404

    # Update the 'modifiedAmount' field in the transaction document
    transaction["modified_amount"] = new_amount

    # Save the updated document back to the database
    success = mongo.update_transaction(transaction_id, transaction)

    if success:
        return jsonify({"message": "Transaction amount updated successfully."})
    else:
        return jsonify({"error": "Failed to update transaction amount."}), 500


@prod_bp.route("/api/prod/toggle-transaction/<transaction_id>", methods=["PUT"])
def toggle_transaction(transaction_id):
    # Use the transaction_id to find the transaction in the database
    transaction = mongo.find_transaction_by_id(transaction_id)

    if transaction is None:
        return jsonify({"error": "Transaction not found."}), 404

    # Check if 'is_hidden' field exists in the transaction document
    if "is_hidden" in transaction:
        # If it exists, toggle its value (from True to False or vice versa)
        transaction["is_hidden"] = not transaction["is_hidden"]
    else:
        # If it doesn't exist, set it to True
        transaction["is_hidden"] = True

    # Save the updated document back to the database
    success = mongo.update_transaction(transaction_id, transaction)

    if success:
        return jsonify(
            {"message": "Transaction 'is_hidden' field toggled successfully."}
        )
    else:
        return jsonify({"error": "Failed to toggle 'is_hidden' field."}), 500


# @insights_bp.route("/api/insights/spending/monthly", methods=["GET"])
# def get_daily_spending():
#     date_str = request.args.get("date")
#     date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")

#     start_date = date
#     end_date = date + timedelta(days=1)

#     daily_transactions = filter_transactions(transaction_data, start_date, end_date)

#     total_spending = sum(transaction["amount"] for transaction in daily_transactions)

#     return jsonify({"date": date_str, "total_spending": total_spending})


# @insights_bp.route("/api/insights/spending/weekly", methods=["GET"])
# def get_weekly_spending():
#     date_str = request.args.get("date")
#     date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")

#     start_date = date - timedelta(days=date.weekday())  # Start of the week (Monday)
#     end_date = start_date + timedelta(days=7)

#     weekly_transactions = filter_transactions(transaction_data, start_date, end_date)

#     total_spending = sum(transaction["amount"] for transaction in weekly_transactions)

#     return jsonify(
#         {
#             "start_date": start_date.strftime("%a, %d %b %Y"),
#             "end_date": end_date.strftime("%a, %d %b %Y"),
#             "total_spending": total_spending,
#         }
#     )


# # Endpoint to get monthly spending
# @insights_bp.route("/api/insights/spending/monthly", methods=["GET"])
# def get_monthly_spending():
#     date_str = request.args.get("date")
#     date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")

#     start_date = date.replace(day=1)
#     if start_date.month == 12:
#         end_date = start_date.replace(year=start_date.year + 1, month=1)
#     else:
#         end_date = start_date.replace(month=start_date.month + 1)

#     monthly_transactions = filter_transactions(transaction_data, start_date, end_date)

#     total_spending = sum(transaction["amount"] for transaction in monthly_transactions)

#     return jsonify(
#         {
#             "start_date": start_date.strftime("%a, %d %b %Y"),
#             "end_date": end_date.strftime("%a, %d %b %Y"),
#             "total_spending": total_spending,
#         }
#     )


@prod_bp.route("/api/prod/insights/transactions/monthly", methods=["GET"])
def get_all_monthly_transactions():
    date_str = request.args.get("date")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify(
            {"error": "Invalid date format. Use the format: 'YYYY-MM-DD'"}, 400
        )

    # Fetch all transactions from MongoDB and convert date strings to datetime objects
    all_transactions = mongo.get_all_transactions()

    for transaction in all_transactions:
        transaction["date"] = datetime.strptime(
            transaction["date"], "%a, %d %b %Y %H:%M:%S %Z"
        )

    # Calculate the start and end dates for the given month
    start_date = date.replace(day=1)
    if start_date.month == 12:
        end_date = start_date.replace(year=start_date.year + 1, month=1)
    else:
        end_date = start_date.replace(month=start_date.month + 1)

    monthly_transactions = filter_transactions(all_transactions, start_date, end_date)

    return jsonify(monthly_transactions)


@prod_bp.route("/api/prod/insights/transactions/last-6-months", methods=["GET"])
def get_last_6_monthly_spending():
    date_str = request.args.get("date")

    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify(
            {"error": "Invalid date format. Use the format: 'YYYY-MM-DD'"}, 400
        )

    # Fetch all transactions from MongoDB
    all_transactions = mongo.get_all_transactions()
    for transaction in all_transactions:
        transaction["date"] = datetime.strptime(
            transaction["date"], "%a, %d %b %Y %H:%M:%S %Z"
        )

    # Initialize a list to store the aggregated spending for the last 6 months
    aggregated_data = []

    date = date.replace(month=date.month+1)

    for _ in range(6):
        end_date = date.replace(day=1)
        
        if end_date.month == 1:
            start_date = end_date.replace(year=end_date.year - 1, month=12)
        else:
            start_date = end_date.replace(month=end_date.month - 1)

        # Filter transactions that fall within the date range
        monthly_transactions = [
            transaction
            for transaction in all_transactions
            if (
                start_date <= transaction["date"] <= end_date
                and transaction["amount"] > 0
                and (
                    "is_hidden" not in transaction
                    or ("is_hidden" in transaction and not transaction["is_hidden"])
                )
            )
        ]

        total_spending = 0

        for transaction in monthly_transactions:
            if (
                "modified_amount" in transaction
                and transaction["amount"] != transaction["modified_amount"]
            ):
                total_spending += transaction["modified_amount"]
            else:
                total_spending += transaction["amount"]

        aggregated_data.append(
            {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "total_spending": round(total_spending, 2),
            }
        )
        # Update the date to the previous month
        date = start_date

    # Reverse the list to have the data in chronological order
    aggregated_data.reverse()

    return jsonify(aggregated_data)


@prod_bp.route("/api/prod/insights/report", methods=["GET"])
def get_monthly_insight_report():
    date_str = request.args.get("date")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify(
            {"error": "Invalid date format. Use the format: 'YYYY-MM-DD'"}, 400
        )

    # Fetch all transactions from MongoDB and convert date strings to datetime objects
    all_transactions = mongo.get_all_transactions()

    for transaction in all_transactions:
        transaction["date"] = datetime.strptime(
            transaction["date"], "%a, %d %b %Y %H:%M:%S %Z"
        )

    # Calculate the start and end dates for the given month
    start_date = date.replace(day=1)
    if start_date.month == 12:
        end_date = start_date.replace(year=start_date.year + 1, month=1)
    else:
        end_date = start_date.replace(month=start_date.month + 1)

    monthly_transactions = filter_transactions(all_transactions, start_date, end_date)

    monthly_transactions_to_analyze = []

    for transaction in monthly_transactions:
        data_to_append = {}
        data_to_append["amount"] = transaction["amount"]
        data_to_append["category"] = transaction["personal_finance_category"]["primary"]
        data_to_append["name"] = transaction["name"]
        monthly_transactions_to_analyze.append(data_to_append)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant providing personal financial advice. You want to give the best tips for a person to help improve their money management and build their net worth over time. I want it to be brief and give the user good financial advice as well. Try keep it concise. Ideally, bullet point styles, to the point but informative.",
            },
            {
                "role": "user",
                "content": "Give me things I'm doing well and three things im not in terms of financial advice: "
                + str(monthly_transactions_to_analyze),
            },
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0.05,
        presence_penalty=0,
    )

    return jsonify(response)


@prod_bp.route("/api/prod/insights/card", methods=["GET"])
def get_card_recommendations():
    date_str = request.args.get("date")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify(
            {"error": "Invalid date format. Use the format: 'YYYY-MM-DD'"}, 400
        )

    # Fetch all transactions from MongoDB and convert date strings to datetime objects
    all_transactions = mongo.get_all_transactions()

    for transaction in all_transactions:
        transaction["date"] = datetime.strptime(
            transaction["date"], "%a, %d %b %Y %H:%M:%S %Z"
        )

    # Calculate the start and end dates for the given month
    start_date = date.replace(day=1)
    if start_date.month == 12:
        end_date = start_date.replace(year=start_date.year + 1, month=1)
    else:
        end_date = start_date.replace(month=start_date.month + 1)

    monthly_transactions = filter_transactions(all_transactions, start_date, end_date)

    monthly_transactions_to_analyze = []

    for transaction in monthly_transactions:
        data_to_append = {}
        data_to_append["amount"] = transaction["amount"]
        data_to_append["category"] = transaction["personal_finance_category"]["primary"]
        data_to_append["name"] = transaction["name"]
        monthly_transactions_to_analyze.append(data_to_append)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    test = "Based on your spending, here are the recommended credit and/or debit cards:\n\n1. Capital One Savor Cash Rewards Credit Card: This card offers unlimited 4% cash back on dining and entertainment, making it a great option for your food and drink expenses. You can also earn 2% cash back at grocery stores and 1% cash back on all other purchases.\n\n2. Uber Visa Card: Since you frequently use Uber for transportation, this card provides 5% back in Uber Cash for Uber rides, Uber Eats, and JUMP bikes and scooters. You also earn 3% cash back on restaurants, bars, and delivery services.\n\nThese cards will help you maximize rewards and discounts on your most frequent spending categories. Make sure to review the terms and conditions of each card to determine which one aligns best with your needs."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant providing personal financial advice. You are trying to help a user making better financial decisions. In this case, you are trying to recommend 2 to 4 credit and/or debit cards based on the person's spendings either to maximize points or rewards or get the most discounts possible. Additionally, the results should be given by name: and the reason:. Do not number the list of cards.",
            },
            {
                "role": "user",
                "content": 'Recommend me best credit and/or debit cards based on my spending:\n\n[\n  {\n    "amount": 152.66,\n    "category": "FOOD_AND_DRINK",\n    "name": "South End Buttery, Toast.inc"\n  },\n  {\n    "amount": 25.21,\n    "category": "TRANSPORTATION",\n    "name": "Uber 072515 SF**POOL**"\n  },\n  {\n    "amount": 74.26,\n    "category": "GENERAL_MERCHANDISE",\n    "name": "TARGET"\n  },\n  {\n    "amount": 13.72,\n    "category": "GENERAL_MERCHANDISE",\n    "name": "Amazon.com"\n  },\n  {\n    "amount": 25.0,\n    "category": "TRAVEL",\n    "name": "MBTA"\n  },\n  {\n    "amount": 12.33,\n    "category": "TRANSPORTATION",\n    "name": "Uber 072515 SF**POOL**"\n  },\n  {\n    "amount": 4.12,\n    "category": "FOOD_AND_DRINK",\n    "name": "Tatte"\n  },\n  {\n    "amount": 4.55,\n    "category": "FOOD_AND_DRINK",\n    "name": "Starbucks"\n  },\n  {\n    "amount": 17.8,\n    "category": "FOOD_AND_DRINK",\n    "name": "KFC"\n  },\n  {\n    "amount": 12.5,\n    "category": "FOOD_AND_DRINK",\n    "name": "Santouka Back Bay"\n  }\n]',
            },
            {
                "role": "assistant",
                "content": "Based on your spendings, here are the recommended credit and/or debit cards:\n\n\nChase Freedom Unlimited: You can earn unlimited 1.5% cash back on all purchases, including your spendings on food and drink, transportation, general merchandise, and more.\n\nCapital One VentureOne Rewards: This card offers 1.25 miles per dollar on every purchase, making it a great option for your spending on transportation and travel.\n\nAmazon Prime Rewards Visa Signature Card: Since you frequently shop on Amazon.com, this card gives you 5% back on all your Amazon purchases, helping you save money.\n\nTarget REDcard: If you shop frequently at Target, this card offers you a 5% discount on all Target purchases. It can help you save money on your general merchandise spending.\n\nConsider applying for these cards based on your specific needs to maximize rewards and discounts.",
            },
            {
                "role": "user",
                "content": "Recommend me best credit and/or debit cards based on my spending:\n "
                + str(monthly_transactions_to_analyze),
            },
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    test = test.split("\n\n")

    return jsonify(response["choices"][0]["message"]["content"])


@prod_bp.route("/api/prod/fetch/card", methods=["GET"])
def fetch_card_rec():
    content = [
        {"title": "Uber Visa Card",
        "reason": "This card is ideal for your transportation expenses, including Uber rides. It offers cash back on Uber rides, as well as on other spending categories, such as restaurants."},

        {"title": "Capital One Walmart Rewards Card",
        "reason": "If you frequently shop at Walmart, this card can provide you with rewards and discounts on your general merchandise purchases. It offers cash back on Walmart purchases as well as on other eligible spending."},

        {"title": "Amazon Prime Rewards Visa Signature Card",
        "reason": "Since you have made purchases on Amazon.com, this card can earn you rewards and discounts on Amazon purchases. It offers cash back on Amazon purchases, as well as on other spending categories."},

        {"title": "American Express Gold Card",
        "reason": "If you spend a significant amount on food and drink, this card can be beneficial. It offers rewards and benefits at restaurants worldwide, including fast-food establishments."}
    ]

    return jsonify(content)
