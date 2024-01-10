import openai
import configparser

# Load or create the configuration file
config = configparser.ConfigParser()

try:
    config.read("config.ini")
    api_key = config.get("OpenAI", "api_key")
except (configparser.NoSectionError, configparser.NoOptionError):
    api_key = None

# If API key is not found or is invalid, prompt the user for a new one
while not api_key:
    api_key = input("Enter your OpenAI API key: ")

    # Save the API key to the configuration file
    config.add_section("OpenAI")
    config.set("OpenAI", "api_key", api_key)
    
    with open("config.ini", "w") as configfile:
        config.write(configfile)

# Set the API key for the OpenAI library
openai.api_key = api_key

def chatbot(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I encountered an error. Please try again."

if __name__ == "__main__":
    print("Welcome to AIED_Chatbot. Type 'quit', 'exit', or 'bye' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response = chatbot(user_input)
        print(f"AIED_Chatbot: {response}")
