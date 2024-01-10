# main.py

import streamlit as st
import os
import json
import requests
import openai
import time
from dotenv import load_dotenv

load_dotenv()

# Replace with the actual method or variable to retrieve your API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")    

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["FMP_API_KEY"] = FMP_API_KEY

# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["FMP_API_KEY"] = "11ArDcx4kTFOFxUd1P10eLe5awIL83yv"
# # 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
# api_key = "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"
# FMP_API_KEY = "11ArDcx4kTFOFxUd1P10eLe5awIL83yv"


# Initialize OpenAI client
# client = openai.OpenAI()

def initialize_openai_client(OPENAI_API_KEY):    
    return openai.OpenAI(api_key=OPENAI_API_KEY)
    
client = openai.OpenAI(api_key=OPENAI_API_KEY)
# Define financial statement functions (same as in the notebook)

# Step 1: Defining Financial Functions

# Define financial statement functions
def get_income_statement(ticker, period, limit):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    return json.dumps(response.json())

def get_balance_sheet(ticker, period, limit):
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    return json.dumps(response.json())

def get_cash_flow_statement(ticker, period, limit):
    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    return json.dumps(response.json())

def get_key_metrics(ticker, period, limit):
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    return json.dumps(response.json())

def get_financial_ratios(ticker, period, limit):
    url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    return json.dumps(response.json())

def get_financial_growth(ticker, period, limit):
    url = f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    return json.dumps(response.json())


# Map available functions (same as in the notebook)

# Step 2: Map available functions

# Map available functions
available_functions = {
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_cash_flow_statement": get_cash_flow_statement,
    "get_key_metrics": get_key_metrics,
    "get_financial_ratios": get_financial_ratios,
    "get_financial_growth": get_financial_growth
}


# Creating an assistant with specific instructions and tools (same as in the notebook)
assistant = client.beta.assistants.create(
    instructions="Act as a financial analyst by accessing detailed financial data through the Financial Modeling Prep API. Your capabilities include analyzing key metrics, comprehensive financial statements, vital financial ratios, and tracking financial growth trends.",
    model="gpt-3.5-turbo-1106",
    tools=[
        {"type": "function", "function": {"name": "get_income_statement", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string"}, "limit": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_balance_sheet", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string"}, "limit": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_cash_flow_statement", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string"}, "limit": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_key_metrics", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string"}, "limit": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_financial_ratios", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string"}, "limit": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_financial_growth", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string"}, "limit": {"type": "integer"}}}}},
    ])

# Creating a new thread
thread = client.beta.threads.create()

# Streamlit UI
st.title("Financial Analyst Assistant")

# User input for the prompt
user_input = st.text_input("User Prompt:")

client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_input
)
# Display user input
st.text(f"User: {user_input}")

# Run Assistant button
if st.button("Run Assistant"):
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

    # Monitor and Manage the Run
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        # Display run steps
        run_steps = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id)
        print(f"Run Steps: {run_steps}")

        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    output = function_to_call(**function_args)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output,
                    })

            # Submit tool outputs and update the run
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        elif run.status == "completed":
            # Display assistant response
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for message in messages.data:
                role_label = "User" if message.role == "user" else "Assistant"
                message_content = message.content[0].text.value
                st.text(f"{role_label}: {message_content}")
            break  # Exit the loop after processing the completed run

        elif run.status == "failed":
            st.text("Run failed.")
            break

        elif run.status in ["in_progress", "queued"]:
            st.text(f"Run is {run.status}. Waiting...")
            time.sleep(5)  # Wait for 5 seconds before checking again

        else:
            st.text(f"Unexpected status: {run.status}")
            break
