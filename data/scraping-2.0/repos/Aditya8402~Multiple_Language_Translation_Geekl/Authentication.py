import openai

# Function to check if given API key is correct or not.

def check_key(api_key):
    openai.api_key = api_key

    try:
        reply = openai.ChatCompletion.create(           # Function to check if gpt is responding with a reply or not.
            model = "gpt-3.5-turbo",
            messages = [
                {"role":"system", "content":"Hello"}
            ],
            temperature = 0.2,
        )
        return True
        
    except:
        return False