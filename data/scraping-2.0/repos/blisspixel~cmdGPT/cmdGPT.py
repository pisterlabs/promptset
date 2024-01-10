import os
import argparse
import asyncio
import openai
from datetime import datetime
from dotenv import load_dotenv
from colorama import Fore, init

from utils import (clear_screen, display_initial_title, display_short_title, 
                   animate_processing, clear_processing_message, sanitize_for_filename, check_and_run_getvoices)
from logging_config import setup_logging
from voice_handler import select_voice, load_custom_voices, stream_audio_websocket
from chat_management import save_chat_transcript, save_audio_file, manage_audio_files
from api_interaction import interact_with_model

# Initialize colorama and load environment variables, set up logging
init(autoreset=True)
load_dotenv()
setup_logging()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define colors for different types of messages
user_color = Fore.RED
cmdGPT_color = Fore.WHITE
system_color = Fore.LIGHTBLACK_EX

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="cmdGPT Chat Application")
    parser.add_argument('--model', type=str, default=None, help='Choose the model to use')
    parser.add_argument('--voice', type=int, default=None, help='Choose the voice option')
    parser.add_argument('--system', type=str, default=None, help='Specify a system message')
    return parser.parse_args()

def select_model():
    """Select the GPT model to use."""
    print("\nSelect a model:")
    models = {
        "1": "gpt-4-1106-preview",
        "2": "gpt-4-vision-preview",
        "3": "gpt-3.5-turbo-1106",
        "4": "gpt-3.5-turbo"
    }
    for key, value in models.items():
        print(f"{key}. {value}")
    choice = input("Enter your choice (default is 1): ")
    return models.get(choice, "gpt-4-1106-preview")

async def chat():
    check_and_run_getvoices()
    args = parse_args()

    while True:
        display_initial_title()

        model = args.model if args.model else select_model()
        custom_voices = load_custom_voices()

        # Handling voice configuration
        if args.voice is not None:
            try:
                voice_config = custom_voices[args.voice - 1]  # Adjusted for zero-based indexing
            except IndexError:
                print("Invalid voice option. Defaulting to No Voice.")
                voice_config = None
        else:
            voice_config = select_voice()

        # Handling system message
        system_message = args.system if args.system else input(f"\n{system_color}Enter a system message or press Enter for default: ")
        if not system_message:
            system_message = "You are a helpful assistant who responds very accurately, VERY concisely, and intelligently. Respond with an element of reddit/4chan humor but keep it professional."

        clear_screen()
        display_short_title(model, voice_config['name'] if voice_config else None, system_message)
        messages = [{"role": "system", "content": system_message}]
        last_saved_index = 0  # Initialize the index for the last saved message

        while True:
            user_input = input(f"\n{user_color}You: ")
            if user_input.lower() in ["exit", "quit"]:
                save_chat_transcript(messages, last_saved_index)
                return
            elif user_input.lower() == "reset":
                save_chat_transcript(messages, last_saved_index)
                break  # Breaks the inner loop, causing the outer loop to restart the selections
            elif user_input.lower() == "clear":
                save_chat_transcript(messages, last_saved_index)
                messages = [{"role": "system", "content": system_message}]
                last_saved_index = len(messages)  # Resets the conversation while keeping current settings
                clear_screen()
                display_short_title(model, voice_config['name'] if voice_config else None, system_message)
                continue

            messages.append({"role": "user", "content": user_input})

            processing_task = asyncio.create_task(animate_processing(f"{system_color}Processing OpenAI Chat"))
            response = await asyncio.to_thread(interact_with_model, model, messages)
            processing_task.cancel()
            clear_processing_message()

            if response:
                if voice_config:
                    audio_processing_task = asyncio.create_task(animate_processing(f"{system_color}Processing Audio"))

                    def stop_audio_animation():
                        audio_processing_task.cancel()
                        clear_processing_message()

                    audio_buffer = await stream_audio_websocket(voice_config, response, stop_audio_animation)

                    response_filename = sanitize_for_filename(response)
                    audio_filename = f"{response_filename}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
                    save_audio_file(audio_buffer, audio_filename)
                    manage_audio_files()

                    print(f"{cmdGPT_color}cmdGPT: {response}")
                else:
                    print(f"{cmdGPT_color}cmdGPT: {response}")

                messages.append({"role": "assistant", "content": response})
                save_chat_transcript(messages, last_saved_index)
                last_saved_index = len(messages)  # Update the index for the last saved message

if __name__ == "__main__":
    asyncio.run(chat())
