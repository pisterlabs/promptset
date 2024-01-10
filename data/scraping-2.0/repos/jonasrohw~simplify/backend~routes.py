from flask import request, jsonify, make_response
import openai
from app import app, db
from models import Transaction,Message,Contact
from utils import get_all_transactions
from enum import Enum
from sqlalchemy import asc
import random
import json


class HistoryTypes(Enum):
    user = 'user'
    system = 'system'

@app.route('/message', methods=['POST', 'OPTIONS'])
def get_bot_reply():
    if request.method == 'OPTIONS':
        app.logger.info('Received OPTIONS request')
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    data = request.json
    user_message_content = data['userMessage']

    messages = [{"role": "system", "content": "You are a helpful banking app assistant."}]

    # Retrieve all previous messages from the database and add them to the list
    saved_messages = Message.query.order_by(Message.id.asc()).all()
    for msg in saved_messages:
        role = "user" if msg.type == 'user' else "system"  # Update as per your HistoryTypes model
        messages.append({"role": role, "content": msg.message})

    # Add the current user message
    messages.append({"role": "user", "content": user_message_content})

    # Save user message to the database
    user_message = Message(message=user_message_content, type='user')
    db.session.add(user_message)
    db.session.commit()

    functions = [
        {
            "name": "get_all_transactions",
            "description": "Retrieve all banking transactions",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "create_contact",
            "description": "Create a new contact with a name and IBAN",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "IBAN": {"type": "string"}
                },
                "required": ["name", "IBAN"]
            }
        },
        {
        "name": "visualize_transactions",
        "description": "Visualize banking transactions for the last N days",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer"}
            },
            "required": ["days"]
        }
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_message = response["choices"][0]["message"]
   

    # Check for function calls
    if "function_call" in response_message:
        function_name = response_message["function_call"]["name"]

        if function_name == "get_all_transactions":
            # Add your implementation of get_all_transactions here
            function_response = "Transactions retrieved successfully."  # Example response
            bot_reply_content = function_response
            bot_reply_type = "message"

        elif function_name == "create_contact":
            function_params = response_message["function_call"].get("parameters", {})
            name = function_params.get("name")
            IBAN = function_params.get("IBAN")
            # Add your implementation of create_contact here
            function_response = f"Contact {name} with IBAN {IBAN} created successfully."  # Example response
            bot_reply_content = function_response
            bot_reply_type = "message"
        elif function_name == "visualize_transactions":
            function_params = response_message["function_call"]
            arguments_str = function_params.get("arguments", "{}")
            try:
                arguments_dict = json.loads(arguments_str)
            except json.JSONDecodeError:
                print("Error: The 'arguments' field is not a valid JSON string.")
                # Handle the error appropriately, e.g., set a default value or return an error response

            # Extract and convert the 'days' value to an integer
            days = arguments_dict.get("days")
            if isinstance(days, int):
                print("Number of days as int:", days)
            else:
                print("Error: The 'days' field is not an integer.")
    # Handle the error appropriately

            if days is not None and isinstance(days, int) and days > 0:
                # Generate fictional data for the line chart
                labels = [f"Day {i}" for i in range(1, days + 1)]
                expenses_data = [random.uniform(-50, 0) for _ in range(days)]
                profits_data = [random.uniform(0, 50) for _ in range(days)]

                bot_reply_content = {
                    "data": {
                        "labels": labels,
                        "datasets": [
                            {
                                "label": 'Daily Expenses',
                                "data": expenses_data,
                                "fill": False,
                                "borderColor": 'rgb(255, 99, 132)',
                                "tension": 0.1
                            },
                            {
                                "label": 'Daily Profits',
                                "data": profits_data,
                                "fill": False,
                                "borderColor": 'rgb(75, 192, 192)',
                                "tension": 0.1
                            },
                        ]
                    },
                    "options": {
                        "scales": {
                            "y": {
                                "beginAtZero": False
                            }
                        },
                        "responsive": True,
                        "maintainAspectRatio": False
                    }
                }
                bot_reply_type = "line-chart"
            else:
                bot_reply_content = "Invalid number of days provided. Please provide a positive integer."
                bot_reply_type = "message"

    else:
        bot_reply_content = response_message["content"].strip()
        bot_reply_type = "message"
    if bot_reply_type == "message":
        system_message = Message(message=bot_reply_content, type='system')
        db.session.add(system_message)
        db.session.commit()

    response_data = {
        "botReply": {
            "type": bot_reply_type,
            "content": bot_reply_content,
        }
    }
    return jsonify(response_data)


@app.route('/get-transactions', methods=['GET'])
def get_transactions():
    # Query the database for all transactions
    transactions = Transaction.query.all()
    
    # Convert transactions to a list of dictionaries
    output = []
    for transaction in transactions:
        transaction_data = {}
        transaction_data['id'] = transaction.id
        transaction_data['date'] = transaction.date
        transaction_data['amount'] = transaction.amount
        transaction_data['category'] = transaction.category
        output.append(transaction_data)
    
    return output



@app.route('/get-messages', methods=['GET'])
def get_messages():
    # Query the database for all history entries
    history_entries = Message.query.all()
    
    # Convert history entries to a list of dictionaries
    output = []
    for entry in history_entries:
        entry_data = {}
        entry_data['id'] = entry.id
        entry_data['message'] = entry.message
        entry_data['type'] = entry.type
        output.append(entry_data)
    
    return jsonify(output)

@app.route('/create-message', methods=['POST'])
def create_message():
    data = request.json
    message = data['message']
    type = data['type']

    new_message = Message(message=message, type=type)
    db.session.add(new_message)
    db.session.commit()

    return jsonify(success=True, message="Message entry added successfully!")


@app.route('/delete-messages', methods=['DELETE'])
def delete_messages():
    try:
        Message.query.delete()
        db.session.commit()
        return jsonify(success=True, message="All messages deleted successfully!")
    except Exception as e:
        db.session.rollback()
        return jsonify(success=False, message=str(e)), 500


@app.route('/get-last-transactions', methods=['GET'])
def get_last_transactions(x):
    # Query the database for the last 'x' transactions, ordered by date
    transactions = Transaction.query.order_by(Transaction.date.desc()).limit(x).all()

    # Convert transactions to a list of dictionaries
    output = []
    for transaction in reversed(transactions):  # Reversed to have the oldest transaction first
        transaction_data = {
            'date': transaction.date.strftime('%Y-%m-%d'),  # Format date as string
            'amount': transaction.amount,
            'category': transaction.category
        }
        output.append(transaction_data)

    return output
