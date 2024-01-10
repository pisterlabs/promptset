import openai
import requests

# Set up OpenAI API
openai.api_key = '785a1fe76480d9fbcde3dda93be8fc11'

# External APIs
TRIVIA_API_URL = "https://opentdb.com/api.php?amount=1&type=multiple"
DOG_API_URL = "https://dog.ceo/api/breeds/image/random"

def chat_with_gpt(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=150,
        temperature=0.6,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

def get_weather(city):
    api_key = 'YOUR_WEATHER_API_KEY'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    response = requests.get(url)
    data = response.json()
    if data['cod'] == '404':
        return "City not found. Please try again."
    else:
        weather_desc = data['weather'][0]['description']
        temperature = data['main']['temp'] - 273.15
        return f"The weather in {city} is {weather_desc} with a temperature of {temperature:.2f}°C."

def get_fact():
    response = chat_with_gpt("Tell me an interesting fact")
    return response

def recommend_movie():
    response = chat_with_gpt("Recommend a good movie")
    return response

def get_definition(term):
    response = chat_with_gpt(f"What is the definition of {term}")
    return response

# Function to get a random trivia question
def get_trivia_question():
    response = requests.get(TRIVIA_API_URL)
    data = response.json()
    question = data['results'][0]['question']
    return question

# Function to get a random dog image
def get_random_dog_image():
    response = requests.get(DOG_API_URL)
    data = response.json()
    image_url = data['message']
    return image_url

def main():
    print("~~~ FunChat Virtual Assistant ~~~")
    print("Type 'exit' to quit at any time.")
    user_name = input("Your name, please: ")

    while True:
        user_input = input(f"{user_name}: ")
        if user_input.lower() == 'exit':
            print("Goodbye! See you soon!")
            break
        elif user_input.lower() == 'help':
            print("You can ask questions, get weather, tell jokes, get facts, get definitions, or have a conversation!")
        elif user_input.lower() == 'joke':
            response = chat_with_gpt(f"{user_name}: Tell me a joke")
            print("Assistant:", response)
        elif user_input.lower() == 'fact':
            response = get_fact()
            print("Assistant:", response)
        elif user_input.lower().startswith('weather'):
            city = user_input.split(' ', 1)[1]
            response = get_weather(city)
            print("Assistant:", response)
        elif user_input.lower() == 'calculate':
            expression = input("Your math expression, please: ")
            try:
                result = eval(expression)
                print(f"The result is: {result}")
            except Exception as e:
                print(f"Oops! Something went wrong. Error: {e}")
        elif user_input.lower() == 'recommend':
            response = recommend_movie()
            print("Assistant:", response)
        elif user_input.lower().startswith('define'):
            term = user_input.split(' ', 1)[1]
            response = get_definition(term)
            print("Assistant:", response)
        elif user_input.lower() == 'trivia':
            question = get_trivia_question()
            print("Assistant:", question)
        elif user_input.lower() == 'dog':
            dog_image_url = get_random_dog_image()
            print("Assistant: Here's a random dog image for you:")
            print(dog_image_url)
        else:
            response = chat_with_gpt(f"{user_name}: {user_input}")
            print("Assistant:", response)

if __name__ == "__main__":
    main()
