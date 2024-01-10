import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatCompletionHandler:
    def __init__(self, emotion_analysis):
        self.test_dataset_with_prompt = emotion_analysis.test_dataset_with_prompt
        self.low_temperature = 0.2
        self.high_temperature = 0.8

    def chat_completion(self, fine_tuned_model, messages, temperature):
        print("\033[0m" + "Starting chat_completion...")
        try:
            completion = client.chat.completions.create(
                model=fine_tuned_model,
                messages=messages,
                temperature=temperature
            )

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

        print("\033[92m\u2714 " + f'Completion with temperature {temperature}: ', completion.choices[0].message.content)
        return completion

    def convert_to_serializable(self, item):
        try:
            if isinstance(item, set):
                return list(item)
            else:
                return item
        
        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None

    def generate_chat_completion(self, fine_tuned_model, test_dataset_with_prompt):
        print("\033[0m" + "Starting generate_chat_completion...")
        try:
            generated_chat_completion = []
            with open(test_dataset_with_prompt, "r") as test_data:
                for line in test_data:
                    data = json.loads(line)
                    data_message = data.get("messages")
                    user_review = data_message[1].get("content")
                    print("\033[0m" + 'User review: ', user_review)

                    completion_low_temperature = self.chat_completion(fine_tuned_model, data_message, self.low_temperature)
                    if completion_low_temperature is None:
                        print("\033[91m\u2718 " + f"Error in chat completion with temperature '{self.low_temperature}'")
                        return None
                    completion_high_temperature = self.chat_completion(fine_tuned_model, data_message, self.high_temperature)
                    if completion_high_temperature is None:
                        print("\033[91m\u2718 " + f"Error in chat completion with temperature '{self.high_temperature}'")
                        return None

                    result = [user_review, completion_low_temperature.choices[0].message.content, completion_high_temperature.choices[0].message.content]
                    generated_chat_completion.append(result)
            
            generated_chat_completion = [self.convert_to_serializable(item) for item in generated_chat_completion]
            if generated_chat_completion is None:
                print("\033[91m\u2718 " + "Error in generate_chat_completion")
                return None

            return generated_chat_completion

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None