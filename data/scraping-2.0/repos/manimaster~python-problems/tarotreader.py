import openai

class TarotReader:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_reading(self, question):
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=f"Tarot reading for: {question}\n\nReading:",
            max_tokens=150
        )
        return response.choices[0].text.strip()

def main():
    api_key = input("Enter your OpenAI API key: ").strip()
    reader = TarotReader(api_key)

    while True:
        question = input("Enter your question for Tarot reading (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        reading = reader.get_reading(question)
        print("\nTarot Reading:", reading, "\n")

if __name__ == "__main__":
    main()
