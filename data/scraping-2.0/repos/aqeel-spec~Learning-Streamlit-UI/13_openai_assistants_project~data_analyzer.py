from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads.thread_message import ThreadMessage
from openai.types.beta.threads.run import Run
import json
from dotenv import load_dotenv, find_dotenv
import time
from typing import Any
import requests
from PIL import Image
from IPython.display import display
import os

class MessageItem:
    def __init__(self, role: str, content: str | Any):
        self.role: str = role
        self.content: str | Any = content

# function_schema = 
# Define financial statement functions






class DataAnalyzerBot:

    def __init__(self, name:str, instructions:str, model:str = "gpt-3.5-turbo-1106")->None:
        self.name: str = name
        self.instructions: str = instructions
        self.model: str = model
        # self.OPENAI_API_KEY : str = "",
        # self.FMP_API_KEY : str = "",
        load_dotenv(find_dotenv())
        self.client : OpenAI = OpenAI()
        # get FMP_API_KEY
        self.FMP_API_KEY = os.getenv("FMP_API_KEY")
        


        self.assistant: Assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions= self.instructions,
            model=self.model,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_income_statement",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                            },
                            "required": ["ticker"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_key_metrics",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                            },
                            "required": ["ticker"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_financial_ratios",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                            },
                            "required": ["ticker"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_financial_growth",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                            },
                            "required": ["ticker"],
                        },
                    },
                },
            ],
        )
        self.thread: Thread  = self.client.beta.threads.create()
        self.messages: list[MessageItem] = []
    def get_income_statement(self,ticker : str) -> str:
        # key_value = self.FMP_API_KEY[0]  # Assuming self.FMP_API_KEY is a tuple
        # print("get_income_statement key_value :",key_value)
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&apikey=doWQxGoRbXisQ9F6zwGSMc1gxC3M7P0Z"
        response = requests.get(url)
        return json.dumps(response.json())

    def get_balance_sheet(self, ticker: str) -> str:
        # key_value = self.FMP_API_KEY[0]
        # print("get_balance_sheet key_value :",key_value)
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=annual&apikey=doWQxGoRbXisQ9F6zwGSMc1gxC3M7P0Z"
        response = requests.get(url)
        return json.dumps(response.json())


    def get_cash_flow_statement(self ,ticker: str) -> str:
        # key_value = self.FMP_API_KEY[0]
        # print("get_cash_flow_statement key_value :",key_value)
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period=annual&apikey=doWQxGoRbXisQ9F6zwGSMc1gxC3M7P0Z"
        response = requests.get(url)
        return json.dumps(response.json())


    def get_key_metrics(self , ticker: str) -> str:
        # key_value = self.FMP_API_KEY[0]
        # print("get_key_metrics key_value :",key_value)
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period=annual&apikey=doWQxGoRbXisQ9F6zwGSMc1gxC3M7P0Z"
        response = requests.get(url)
        return json.dumps(response.json())


    def get_financial_ratios(self ,ticker: str) -> str:
        # key_value = self.FMP_API_KEY[0]
        # print("get_financial_ratios key_value :",key_value)
        url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?period=annual&apikey=doWQxGoRbXisQ9F6zwGSMc1gxC3M7P0Z"
        response = requests.get(url)
        return json.dumps(response.json())


    def get_financial_growth(self,ticker: str) -> str:
        # key_value = self.FMP_API_KEY[0]
        # print("get_financial_growth key_value :",key_value)
        url = f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?period=annual&apikey=doWQxGoRbXisQ9F6zwGSMc1gxC3M7P0Z"
        response = requests.get(url)
        return json.dumps(response.json())
    def get_name(self):
        return self.name

    def get_instructions(self):
        return self.instructions

    def get_model(self):
        return self.model
    
    def send_message(self, message: str):
        latest_message: ThreadMessage = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=message
        )

        self.latest_run: Run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=self.instructions
        )

        # print("message sent on thread id: ", self.thread.id)

        self.addMessage(MessageItem(role="user", content=message))
   
    def download_and_save_image(self, file_id: str, save_path: str) -> None:
        download_url = f"https://api.openai.com/v1/files/{file_id}/content"
        response = requests.get(
            download_url, headers={"Authorization": f"Bearer {self.OPENAI_API_KEY}"}
        )
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
        else:
            print(f"Image downloading failed: Status Code {response.status_code}")
    def available_functions(self):
        return {
            "get_income_statement": self.get_income_statement,
            "get_key_metrics": self.get_key_metrics,
            "get_financial_ratios": self.get_cash_flow_statement,
            "get_financial_growth": self.get_financial_ratios
        }
    def isCompleted(self) -> bool:
        print("Status: ", self.latest_run.status)

        while True:
            self.latest_run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.latest_run.id if self.latest_run else None
            )

            if self.latest_run.status == "requires_action":
                tool_calls = self.latest_run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    available_functions = self.available_functions()
                    print("available_functions",available_functions)
                    if function_name in available_functions:
                        print("Calling ", function_name)
                        function_to_call = available_functions[function_name]
                        output = function_to_call(**function_args)
                        print("function result output",output)
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": output,
                            }
                        )
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id, run_id=self.latest_run.id, tool_outputs=tool_outputs
                )
            elif self.latest_run.status == "completed":
                response_messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
                print(f"Run is {self.latest_run.status}.")

                for message in response_messages.data:
                    print("*************")
                    for content in message.content:
                        role_label = "User" if message.role == "user" else "Assistant"
                        if content.type == "text":
                            message_content = content.text.value
                            print(f"{role_label}: {message_content}\n")
                        elif content.type == "image_file":
                            image_file_id = content.image_file.file_id
                            image_save_path = f"output_images/image_{image_file_id}.png"
                            self.download_and_save_image(image_file_id, image_save_path)
                            display(Image(filename=image_save_path))
                
                return True
            elif self.latest_run.status == "failed":
                print("Run failed.")
                return False
            elif self.latest_run.status in ["in_progress", "queued"]:
                print(f"Run is {self.latest_run.status}. Waiting...")
                time.sleep(10)  # Wait for 10 seconds before checking again
            else:
                print(f"Unexpected status: {self.latest_run.status}")
                return False
    def get_lastest_response(self):
        response_messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
        print(f"Run is {self.latest_run.status}.")

        for message in response_messages.data:
            print("*************")
            for content in message.content:
                role_label = "User" if message.role == "user" else "Assistant"
                if content.type == "text":
                    message_content = content.text.value
                    print(f"{role_label}: {message_content}\n")
                elif content.type == "image_file":
                    image_file_id = content.image_file.file_id
                    image_save_path = f"output_images/image_{image_file_id}.png"
                    self.download_and_save_image(image_file_id, image_save_path)
                    display(Image(filename=image_save_path))
        return response_messages
    def getMessages(self)->list[MessageItem]:
        return self.messages

    def addMessage(self, message: MessageItem)->None: 
        self.messages.append(message)

# ... (previous imports and class definitions)

# Define your API keys and other necessary details
OPENAI_API_KEY ="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"
FMP_API_KEY ="doWQxGoRbXisQ9F6zwGSMc1gxC3M7P0Z"

# Create a DataAnalyzerBot object
bot_name = "Data Analyst"
bot_instructions = "Act as a financial advisor by accessing detailed financial data through the Financial Modeling Prep API. Your capabilities include providing an investement advise by analyzing key metrics, comprehensive financial statements, vital financial ratios, and tracking financial growth trends."
bot_model = "gpt-3.5-turbo-1106"

# Create an instance of the DataAnalyzerBot class
data_analyzer_bot = DataAnalyzerBot(
    name=bot_name, 
    instructions=bot_instructions, 
    # ai=OPENAI_API_KEY,
    # fmp=FMP_API_KEY,
    model=bot_model
)
# Define financial statement functions
# def get_income_statement(ticker : str) -> str:
#     url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&apikey={FMP_API_KEY}"
#     response = requests.get(url)
#     return json.dumps(response.json())
# print(get_income_statement("AAPL"))
# print("******************")
# print(data_analyzer_bot.get_income_statement("AAPL"))

# Set the API keys in the class (assuming these are needed)
# data_analyzer_bot.OPENAI_API_KEY = OPENAI_API_KEY
# data_analyzer_bot.FMP_API_KEY = FMP_API_KEY

# Optionally, send a message to the bot
# user_message = "I have a lot of money to invest. Can you suggest if it is better to invest in Apple or Microsoft?"
# data_analyzer_bot.send_message(user_message)


# Run the completion process
# data_analyzer_bot.isCompleted()

# Retrieve the last response or messages
# last_response = data_analyzer_bot.get_lastest_response()
# all_messages = data_analyzer_bot.getMessages()
# print("all_messages",all_messages)
# print("last_response",last_response)
# print(data_analyzer_bot.getMessages())
