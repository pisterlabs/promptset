import openai

class MovieRecommender:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_recommendation(self, input_text):
        prompt = f"Based on the user's preferences: '{input_text}', recommend a movie."
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()

def main():
    api_key = input("Enter your OpenAI API key: ").strip()
    recommender = MovieRecommender(api_key)

    while True:
        input_text = input("Enter a genre or your favorite movies (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break
        recommendation = recommender.get_recommendation(input_text)
        print("\nRecommended Movie:", recommendation, "\n")

if __name__ == "__main__":
    main()
