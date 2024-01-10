import csv
import configparser
import openai

# Load API key from configuration file
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['openai']['api_key']

# Set up your OpenAI API credentials
openai.api_key = api_key

# Load menu items from CSV file
def load_menu_items():
    menu_items = []
    with open('menu_items.csv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            menu_items.append(row)
    return menu_items

# Generate chat response using ChatGPT API
def generate_chat_response(user_input):
    response = openai.Completion.create(
        engine='text-davinci-003',  # Use the appropriate GPT-3 model
        prompt=user_input,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Find menu items matching user query
def find_matching_menu_items(menu_items, user_query):
    matching_items = []
    for item in menu_items:
        if user_query.lower() in item['name'].lower() or user_query.lower() in item['description'].lower():
            matching_items.append(item)
    return matching_items

# Main function to handle user queries and provide recommendations
def handle_user_query(user_input):
    menu_items = load_menu_items()
    matching_items = find_matching_menu_items(menu_items, user_input)
    
    if len(matching_items) > 0:
        # Generate prompt with matching menu items and their descriptions
        prompt = f"You asked: {user_input}\n\n"
        for item in matching_items:
            prompt += f"{item['name']}: {item['description']} - Price: ${item['price']}\n"
    else:
        prompt = user_input
    
    response = generate_chat_response(prompt)
    
    return response

# Example usage
user_query = "What vegetarian options do you have?"
response = handle_user_query(user_query)
print(response)
