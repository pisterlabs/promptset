import openai
from metaphor_python import Metaphor
import API # API.py YOU CAN REPLACE THIS WITH YOUR OWN API KEYS or REPLACE IT DIRECTLY HERE

# Replace with your API keys
openai.api_key = API.openAI_API
metaphor = Metaphor(API.metaphor_API)

# Function to suggest venues using Metaphor
def suggest_event_venues(event_type, location, budget):
    query = f"{event_type} venues in {location} within {budget} budget"
    search_response = metaphor.search(query, use_autoprompt=True)
    return search_response.results[:5]

# Function for chat-based completion
def chat_completion(user_question):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_question},
        ],
    )
    return completion.choices[0].message.content

# Main program
if __name__ == "__main__":
    print("Welcome to the Event Planning Assistant!")

    user_input = input("What type of event are you planning? ")
    event_ideas = chat_completion(f"Generate event ideas for '{user_input}'")

    print("\nHere are some event ideas:")
    print(event_ideas)

    location = input("\nEnter the location for the event: ")
    budget = input("What is your budget for the venue? ")

    venues = suggest_event_venues(user_input, location, budget)

    print("\nTop venue suggestions:")
    for idx, venue in enumerate(venues, start=1):
        venue_name = venue.title
        venue_location = venue.url
        print(f"{idx}. {venue_name}, {venue_location}")

    print("\nThank you for using the Event Planning Assistant!")