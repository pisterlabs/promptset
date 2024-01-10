from openai import OpenAI
import time
from dotenv import load_dotenv
import os
import json
import requests

load_dotenv()


def get_weather_forecast(location: str):
    appid = os.getenv("OPENWEATHER_API_KEY")
    url = f'http://api.weatherapi.com/v1/current.json?q={location}&key={appid}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Extract relevant weather information
            temperature = data["current"]["temp_f"]
            weather_description = data["current"]['condition']["text"]
            humidity = data["current"]["humidity"]

            # Return the weather data
            return {
                "temperature": temperature,
                "description": weather_description,
                "humidity": humidity
            }
        else:
            return {}
    except requests.exceptions.RequestException as e:
        print("Error occurred during API request:", e)


class AssistantManager:
    def __init__(self, api_key: str, model: str = "gpt-4-1106-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None

    def create_assistant(self, name, instructions, tools):
        self.assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model=self.model
        )

    def create_thread(self):
        self.thread = self.client.beta.threads.create()

    def add_message_to_thread(self, role, content):
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role=role,
            content=content
        )

    def run_assistant(self, instructions):
        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions
        )

    def process_messages(self):
        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)

        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")

    def wait_for_completion(self):
        while True:
            time.sleep(5)
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=self.run.id
            )
            print(run_status.model_dump_json(indent=4))

            if run_status.status == 'completed':
                self.process_messages()
                break
            elif run_status.status == 'requires_action':
                print("Function Calling ...")
                self.call_required_functions(run_status.required_action.submit_tool_outputs.model_dump())
            else:
                print("Waiting for the Assistant to process...")

    def call_required_functions(self, required_actions):
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            func_name = action['function']['name']
            arguments = json.loads(action['function']['arguments'])

            if func_name == "get_weather_forecast":
                output = get_weather_forecast(location=arguments['location'])
                print(output)
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output['temperature']
                })
            else:
                raise ValueError(f"Unknown function: {func_name}")

        print("Submitting outputs back to the Assistant...")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs
        )


def main():
    api_key = os.getenv("api_key")
    manager = AssistantManager(api_key)

    # process 1
    manager.create_assistant(
        name="Weather Assistant",
        instructions="You are a personal Weather Assistant",
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
    )
    location = input("Enter your location:")
    # process 2
    manager.create_thread()
    # process 3
    manager.add_message_to_thread(role="user", content=f"What is the weather like in {location}?")
    # process 4
    manager.run_assistant(instructions="Please address the user as Donald Trump.")
    # final
    manager.wait_for_completion()


if __name__ == '__main__':
    main()
