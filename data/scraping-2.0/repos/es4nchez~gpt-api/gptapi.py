import os
import openai
import json
from datetime import datetime
from utils import bcolors as bc
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import subprocess
import tempfile
import re
import base64

class GPT_API:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.models = self.list_models()
        self.audio_mode = False

    def print_models(self):
        print('\n')
        for idx, model_info in enumerate(self.models, start=1):
            print(f"{idx}) - {model_info['description']}")
        print('\n')

    def list_models(self):
        model_descriptions = {
            "GPT-4": "Improvement of GPT-3.5",
            "GPT-3.5": "Understand as well as generate natural language or code",
            "DALL·E": "Generate and edit images given a natural language prompt",
            "Whisper": "Convert audio into text."
        }

        models = openai.models.list()
        latest_models_dict = {"GPT-4": None, "GPT-3.5": None, "DALL·E": None, "Whisper": None}
        latest_models_timestamps = {"GPT-4": 0, "GPT-3.5": 0, "DALL·E": 0, "Whisper": 0}

        # Iterate through the paginated response
        for model in models.data:
            model_id = model.id
            created_timestamp = model.created

            if "gpt-4" in model_id and created_timestamp > latest_models_timestamps["GPT-4"]:
                latest_models_dict["GPT-4"] = model_id
                latest_models_timestamps["GPT-4"] = created_timestamp
            elif "gpt-3.5" in model_id and created_timestamp > latest_models_timestamps["GPT-3.5"]:
                latest_models_dict["GPT-3.5"] = model_id
                latest_models_timestamps["GPT-3.5"] = created_timestamp
            elif "dall-e" in model_id and created_timestamp > latest_models_timestamps["DALL·E"]:
                latest_models_dict["DALL·E"] = model_id
                latest_models_timestamps["DALL·E"] = created_timestamp
            elif "whisper" in model_id and created_timestamp > latest_models_timestamps["Whisper"]:
                latest_models_dict["Whisper"] = model_id
                latest_models_timestamps["Whisper"] = created_timestamp

        formatted_models_list = [
            {"description": f"{category}: {model_descriptions[category]}", "model_name": model_id}
            for category, model_id in latest_models_dict.items()
            if model_id is not None
        ]
        return formatted_models_list


    def generate(self, model_number=None):
        formatted_models_list = self.list_models()

        if model_number is None:
            model_number = input("Enter model number: ")

        try:
            model_number = int(model_number)
        except ValueError:
            print("Invalid input. Please enter a valid model number.")
            return

        model_number -= 1

        if model_number >= len(formatted_models_list) or model_number < 0:
            print(f'Error: model number \"{model_number + 1}\" not found nor available')
            exit(1)
        else:
            model_info = formatted_models_list[model_number]
            self.model = model_info['model_name']
            print(f"Model selected : {model_info['model_name']}")
            if "gpt-4" in model_info['model_name']:
                self.prompt_GPT4()
            elif "gpt-3" in model_info['model_name']:
                self.prompt_GPT3()
            elif "dall" in model_info['model_name']:
                self.prompt_DALLE()


    def print_response_GTP4(self, response):
        print('\n')
        print(f"Role : {bc.YELLOW}{response['role']}{bc.ENDC}")
        print(f"Response : {bc.DARKCYAN}{response['content']} {bc.ENDC}")
        print('\n')

    def print_response_GTP3(self, response):
        print('\n')
        print(f"Response : {response['choices'][0]['text']}")
        print('\n')

    def print_response_DALLE(self, response):
        print('\n')
        # print(f"Full : {response}")
        for idx, item in enumerate(response['data'], start=1):
            print(f"Url {idx}: {item['url']}")
        print('\n')



    def prompt_GPT4(self):
        self.role = input("Choose the role (enter for blank): ")
        prompt = input("Enter the prompt: ")
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": prompt} ] )
        self.print_response_GTP4(completion.choices[0].message)
        while True:
            user_choice = input("Enter for a new prompt, (q) to go back: ")
            if user_choice == 'q':
                return
            elif len(user_choice) == 0:
                self.prompting()
            else:
                print(bc.RED + "----------\nInvalid choice." + bc.ENDC)


    def prompt_GPT3(self):
        prompt = input("Enter the prompt: ")
        completion = openai.Completion.create(
            model=self.model,
            prompt=prompt
        )
        self.print_response_GTP3(completion)
        while True:
            user_choice = input("Enter for a new prompt, (q) to go back: ")
            if user_choice == 'q':
                return
            elif len(user_choice) == 0:
                self.prompting()
            else:
                print(bc.RED + "----------\nInvalid choice." + bc.ENDC)

    def save_images(self, response):
        if not os.path.exists('generated_images'):
            os.makedirs('generated_images')

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_dir = os.path.join('generated_images', date_str)
        os.makedirs(save_dir)

        for idx, item in enumerate(response['data'], start=1):
            url = item['url']
            response = requests.get(url)

            if response.status_code == 200:
                image_path = os.path.join(save_dir, f'image_{idx}.png')
                with open(image_path, 'wb') as file:
                    file.write(response.content)
                print(f'Saved image {idx} to {image_path}')
            else:
                print(f'Failed to retrieve image {idx}: {response.status_code}')

        print('All images saved!\n')

    def prompt_DALLE(self):
        size = 512
        while True:
            input_size = input("Choose the format (" + bc.YELLOW + "256" + bc.ENDC + "|" + bc.YELLOW + "512" + bc.ENDC + "|" + bc.YELLOW + "1024" + bc.ENDC + "), enter for default (512)): ")
            if len(input_size) == 0:
                break
            if int(input_size) in [256, 512, 1024]:
                size = input_size
                break
            else:
                print(bc.RED + "Wrong format" + bc.ENDC)
        input_prompt = input("Enter the prompt: ")
        input_number = input("Enter the number of image to generate (1-10), defaut is (1): ")
        completion =openai.Image.create(
        prompt=input_prompt,
        n=int(input_number),
        size=f"{size}x{size}" )
        self.print_response_DALLE(completion)
        while True:
            input_save = input("Want to save the image(s) (y=yes, n=no): ")
            if input_save == "n" or input_save == "no":
                break
            elif input_save == 'y' or input_save == "yes":
                self.save_images(completion)
                break
            else:
                print(bc.RED + "Wrong answer." + bc.ENDC)
        while True:
            user_choice = input("Enter for a new prompt, (q) to go back: ")
            if user_choice == 'q':
                return
            elif len(user_choice) == 0:
                self.prompting()
            else:
                print(bc.RED + "----------\nInvalid choice." + bc.ENDC)


    def stream_and_play(self, text):
        try:
            response = openai.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=text,
            )
            response.stream_to_file("output.mp3")
            
            subprocess.run(["afplay", "output.mp3"])
            os.remove("output.mp3")
        except Exception as e:
            print("Error occurred while calling OpenAI API:", e)


    def prompt_conversation(self, conversation_history):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=conversation_history,
                stream=True
            )

            audio_content = ""
            for event in response:
                content = event.choices[0].delta.content
                if content:
                    print(content, end='', flush=True)

                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    audio_content += content

                    if audio_content and audio_content[-1] in '.!?:\n':
                        if self.audio_mode:
                            self.stream_and_play(audio_content)
                        audio_content = ""

            if audio_content and self.audio_mode:
                self.stream_and_play(audio_content)

            print('\n')

        except Exception as e:
            print("\nError occurred while calling OpenAI API:", e)
            return "Sorry, I encountered an error."




    def conversation(self):
        print("\nStarting conversation. Type 'quit' to end the conversation.\n")

        conversation_history = []

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("\nEnding conversation.\n")
                break

            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            self.prompt_conversation(conversation_history)


    def prompt_Marvin(self, conversation_history):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=conversation_history,
                stream=True
            )

            audio_content = ""
            for event in response:
                content = event.choices[0].delta.content
                if content:
                    print(content, end='', flush=True)

                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    audio_content += content

                    if audio_content and audio_content[-1] in '.!?:\n':
                        if self.audio_mode:
                            self.stream_and_play(audio_content)
                        audio_content = ""

            if audio_content and self.audio_mode:
                self.stream_and_play(audio_content)

            print('\n')

        except Exception as e:
            print("\nError occurred while calling OpenAI API:", e)
            return "Sorry, I encountered an error."




    def Marvin(self):
        print("\nStarting conversation. Type 'quit' to end the conversation.\n")

        text_file = open("export_pedago_short.txt", "r")
        data = text_file.read()
        text_file.close()

        conversation_history = []
        # conversation_history.append({
        #         "role": "user",
        #         "content": data
        #     })

        while True:
            user_input = input("Ask your question: ")
            if user_input.lower() == 'quit':
                print("\nEnding conversation.\n")
                break
            user_input = data + "\n\nI just sended you the data about the 42 school. i'll now ask something, and you will search the most accurate answer, and say it. dont say anything else. \n" + user_input
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            self.prompt_Marvin(conversation_history)
