import openai
import json
import os
from termcolor import colored
import requests
import random

# Define constants for the LLMs
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.9  # Higher temperature for more varied responses
MAX_TOKENS = 400  # Limit the response length

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

unique_id = random.randint(0, 1000000000)

# Function to generate personalities for the LLMs
def generate_personalities(unique_id):
    try:
        personalities = openai.ChatCompletion.create(
            model=MODEL,
            temperature=0.7,
            max_tokens=500,
            messages=[
                {"role": "system", "content": f"Generate two historical personalities with special skillsets for two different LLMs. We're making agents to solve real world problems. (names, skillsets, personalities, brief description). Consider the random seed to get a unique response: {unique_id}. Use JSON in your response return two objects LLM1 and LLM2."
                }
            ]
        )
        print(personalities)  # Log the API response
    except Exception as e:
        print(f"API call failed: {e}")
        return None, None

    try:
        personalities_content = json.loads(personalities['choices'][0]['message']['content'])
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(personalities['choices'][0]['message']['content'])  # Log the data that failed to parse
        return None, None
    return personalities_content['LLM1'], personalities_content['LLM2']

# Function to generate starting prompts
def generate_starting_prompts(unique_id):
    try:
        prompts = openai.ChatCompletion.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=400,
            messages=[
                {"role": "system", "content": f"Generate three distinct starting prompts for a conversation and return them as a JSON object with keys 'prompt1', 'prompt2', and 'prompt3'. The prompts ought to present a real societal problem that needs solving. Consider the random seed to get a unique response: {unique_id}."}
            ]
        )
        print(prompts)  # Log the API response

        prompts_content = json.loads(prompts['choices'][0]['message']['content'])
    except Exception as e:
        print(f"API call failed: {e}")
        return None

    return prompts_content

# Function to tailor system messages based on personalities (#6 Implementation)
def tailor_system_message(personality, user_message):
    tailored_message = f"{personality['name']}, as a skilled {personality['skillset']} and a {personality['personality']} individual, how would you address the following issue: {user_message}"
    return tailored_message

messages_history = []

def chat_with_llms(llm1, llm2):
    global messages_history
    try:
        # Prepare the system message for LLM1, incorporating the personality
        system_message_llm1 = f"Your task is to embody the following personality: {json.dumps(llm1)}. " \
                              "Respond to the user's question, attempt to solve the problem working with the user. " \
                              "Remember, play your role and keep the conversation going. Be creative! " \
                              "Limit your response to no more than 80 words."
        
        messages_for_llm1 = [{"role": "system", "content": system_message_llm1}] + messages_history
        
        # API call for LLM 1 with constants
        print(colored(f"Sending the following data to the API for LLM1: {MODEL}, {TEMPERATURE}, {MAX_TOKENS}, {messages_for_llm1}", 'cyan'))
        response1 = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages_for_llm1,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Extract LLM1's response content and update messages_history as a "user" message
        response_content_1 = response1['choices'][0]['message']['content']
        messages_history.append({"role": "user", "content": response_content_1})
        
        # Prepare the system message for LLM2, incorporating the personality
        system_message_llm2 = f"Your task is to embody the following personality: {json.dumps(llm2)}. " \
                              "Respond to the user's question, attempt to solve the problem working with the user. " \
                              "Remember, play your role and keep the conversation going. Be creative! " \
                              "Limit your response to no more than 80 words."
        
        # Update messages_for_llm2 to include the response from LLM1
        messages_for_llm2 = [{"role": "system", "content": system_message_llm2}, {"role": "user", "content": response_content_1}]
        
        # API call for LLM 2 with constants
        print(colored(f"Sending the following data to the API for LLM2: {MODEL}, {TEMPERATURE}, {MAX_TOKENS}, {messages_for_llm2}", 'magenta'))
        response2 = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages_for_llm2,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Extract LLM2's response content and update messages_history as a "user" message
        response_content_2 = response2['choices'][0]['message']['content']
        messages_history.append({"role": "user", "content": response_content_2})
        
        # Format and print the responses
        response_content_1_formatted = f"{llm1['name']}: {response_content_1}"
        response_content_2_formatted = f"{llm2['name']}: {response_content_2}"
        print(colored(response_content_1_formatted, 'red'))
        print(colored(response_content_2_formatted, 'blue'))
        
        # Return the responses for consistency in the conversation log
        return response_content_1_formatted, response_content_2_formatted

    except requests.exceptions.Timeout:
        print("Request timed out. Please try again.")
        return None, None
    except openai.error.OpenAIError as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

# Function to save the chat logs
def save_chat_logs(chat_logs):
    with open('chat_logs.json', 'w') as f:
        json.dump(chat_logs, f, indent=4)

# Main function
def main():
    # Initialize chat_logs dictionary
    chat_logs = {
        'personalities': {},
        'starter_prompts': {},
        'selected_prompt': '',
        'conversation': []  # This will store the whole conversation in sequence
    }

    # Generate personalities for the LLMs
    llm1, llm2 = generate_personalities(unique_id)
    if not llm1 or not llm2:
        return  # Exit if we failed to get the personalities

    chat_logs['personalities'] = {'llm1': llm1, 'llm2': llm2}

    # Generate starting prompts
    prompts = generate_starting_prompts(unique_id)
    if not prompts:
        return  # Exit if we failed to get the prompts

    chat_logs['starter_prompts'] = prompts

    # Select a prompt
    print("Select one of the following prompts to start the conversation:")
    for i, prompt in enumerate(prompts.values(), start=1):
        print(f"{i}. {prompt}")
    prompt_index = int(input("Enter the number of the selected prompt: ")) - 1
    selected_prompt = list(prompts.values())[prompt_index]
    chat_logs['selected_prompt'] = selected_prompt
    
    # Start the conversation with the selected prompt
    messages_history.append({"role": "user", "content": selected_prompt})

    while True:
        # Chat with the LLMs
        response_content_1, response_content_2 = chat_with_llms(llm1, llm2)

        if response_content_1 is None or response_content_2 is None:
            retry = input("There was an error processing your request. Would you like to retry? (Y/N): ")
            if retry.lower() not in ['y', 'yes']:
                break
            else:
                continue

        # Update the conversation log with the responses
        chat_logs['conversation'].append({"llm1": response_content_1})
        chat_logs['conversation'].append({"llm2": response_content_2})

        print("Chat logs: ", json.dumps(chat_logs, indent=2))  # Pretty print the chat logs

        # Ask if the user wants to continue or break the loop
        continue_chat = input("Do you want to continue the chat? (Y/N): ")
        if continue_chat.lower() not in ['y', 'yes']:
            save_chat = input("Do you want to save the chat logs? (Y/N): ")
            if save_chat.lower() in ['y', 'yes']:
                save_chat_logs(chat_logs)
            break

if __name__ == "__main__":
    main()
