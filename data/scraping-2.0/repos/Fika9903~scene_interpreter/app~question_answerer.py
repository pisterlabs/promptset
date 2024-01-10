import openai
import json


def print_object_history(arguments_json):
    arguments = json.loads(arguments_json)
    object_history = object_history_json
    object_name = arguments["object_name"]

    if object_name in object_history:
        return json.dumps({"history": object_history[object_name]})
    else:
        return json.dumps({"error": "Object not found"})

class QuestionAnswerer:
    def __init__(self, api_key):
        self.api_key = api_key
        # Define available functions here
        self.available_functions = {"print_object_history": print_object_history}

    

    def print_object_history(self, arguments_json):
        arguments = json.loads(arguments_json)
        object_history = self.object_history_json
        object_name = arguments["object_name"]

        if object_name in object_history:
            return json.dumps({"history": object_history[object_name]})
        else:
            return json.dumps({"error": "Object not found"})

    def answer_question(self, question: str, scene, object_history) -> str:
        scene_context = json.dumps(scene)
        object_history_json = json.dumps(object_history)

        openai.api_key = self.api_key
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "print_object_history",
                    "description": "Retrieve and return the historical data of a specified object.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_name": {
                                "type": "string",
                                "description": "The name of the object for which to retrieve historical data."
                            }
                        },
                        "required": ["object_name"]
                    }
                }
            }
        ]

        messages = [
            {"role": "system", "content": "Your job is to examine a Scene description given in JSON format and answer a question given regarding the scene."},
            {"role": "user", "content": scene_context},
            {"role": "user", "content": question}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        print(response_message)

        if 'tool_calls' in response_message:
            tool_calls = response_message.tool_calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_functions[function_name]

                # Correctly prepare the JSON string with object history and object name for function call
                object_name = json.loads(tool_call.function.arguments)["object_name"]
                function_args = {"object_history": json.loads(object_history_json), "object_name": object_name}
                function_args_json = json.dumps(function_args)
                function_response = function_to_call(arguments_json=function_args_json)

                tool_response_message = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                }

                # Use a default value with next() to avoid StopIteration and handle missing tool_call_index
                tool_call_index = next((i for i, msg in enumerate(messages) if 'tool_calls' in msg and msg['tool_calls'][0]['id'] == tool_call.id), -1)

                if tool_call_index != -1:
                    # Insert the tool response message right after the corresponding tool_calls message
                    messages.insert(tool_call_index + 1, tool_response_message)
                else:
                    # Handling the case where a corresponding 'tool_calls' message is not found
                    print("Corresponding tool call not found for tool call ID:", tool_call.id)

            # Create the second response with the correctly ordered messages
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages
            )
            return second_response.choices[0].message.content.strip()

        else:
            # No tool_calls suggested, use the model's response directly
            return response_message.content.strip()



# Example Usage
# api_key = "YOUR_API_KEY"
# qa = QuestionAnswerer(api_key)
# question = "Where was Brush_0 at 14:00?"
# scene_context = [{"Name": "Brush_0", "Location": {"X": -470, "Y": 350, "Z": 130}}]
# object_history = {'Brush
if __name__ == '__main__':
    print()
