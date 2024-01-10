from rasa_sdk import Action
import cohere


class CohereFallback(Action):
    def name(self):
        return "action_cohere_fallback"

    def run(self, dispatcher, tracker, domain):
        api_key = 'Snm3hpx0paZyQam0QaNzefmHVbw9559kSAna5gdi'  # Replace with your actual Cohere API key
        co = cohere.Client(api_key)

        # Get the user input
        user_input = tracker.latest_message.get('text')

        # Generate text based on user input
        response = co.generate(
            model='command-nightly',
            prompt=user_input+'-Walchand College of engineerin,Sangli'
            ,
            max_tokens=200,
            temperature=0.750
        )

        # Access the generated text
        generated_response = response.generations[0].text

        # Send the generated response as a fallback
        dispatcher.utter_message(text=generated_response)
        return []
