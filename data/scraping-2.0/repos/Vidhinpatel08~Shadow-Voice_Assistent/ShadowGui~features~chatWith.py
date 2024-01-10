import openai #  pip install openai
import features.TTS as TTS


# Initialize the OpenAI API client
# openai.api_key = ""

# Define a function to generate a response from the GPT-3 model
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"].strip()

# Define a function to handle the conversation with the user
def chat_with_AI(query):
    try:
        response = generate_response(query)
        # TTS.speak_Print(f'\n{response[:500]}')
        TTS.speak_Print(f'\n{response}')
    except Exception as e : 
        TTS.speak("Sorry, I couldn't find an answer. Please try asking again.")

if __name__ == '__main__':
    # Start the conversation
    chat_with_AI("what is Database")