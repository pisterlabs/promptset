import openai
import os
import json

def generate_notification_completion(current_date, script_dir, output_dir):
    
    print("Starting the generate_chat_completion function...")
    
    # Read the system prompt from file
    notificationPromptFilePath = os.path.join(script_dir, 'notification_generator', 'notificationPrompt.txt')
    with open(notificationPromptFilePath, 'r') as file:
        systemPrompt = file.read()
    
    print("System prompt read from file.")

    # Load the existing menu.json file
    with open(f'{output_dir}/{current_date}/menu.json', 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
    
    print("Menu data loaded from JSON file.")

    # Convert the menu data to a string
    user_message = json.dumps(menu_data)
    
    print("Menu data converted to string.")

    # Create a chat completion with the menu data as the content of the user message
    for attempt in range(3):
        print(f"Attempt {attempt+1} to create chat completion...")
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                { "role": "system", "content": systemPrompt },
                {"role": "user", "content": user_message}
            ]
        )

        # Check if the output format is correct
        try:
            output = json.loads(completion.choices[0].message['content'])
            if 'notification' in output and 'title' in output['notification'] and 'body' in output['notification']:
                print("Output format is correct.")
                break
        except json.JSONDecodeError:
            print("JSONDecodeError occurred. Retrying...")
            pass

    # Write the chat completion to the notification.json file
    with open(f'{output_dir}/{current_date}/notification.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"Response added to {current_date}/notification.json")

# Example usage:
# generate_chat_completion("2023-09-19", "/Users/linus/intern-projects/Uni/MIM/PWA23/MensaPWA/nightly_task", "/Users/linus/intern-projects/Uni/MIM/PWA23/MensaPWA/nightly_task/out")
