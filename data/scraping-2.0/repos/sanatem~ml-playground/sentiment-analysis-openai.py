from openai import OpenAI

def get_sentiment(text):
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    max_tokens=10,
    messages=[
        {"role": "system", "content": "You are a sentiment analysis model." +
         "Your responses are ONLY a score from -1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive."},
        {"role": "user", "content": "Analyze the following text: " + text},
    ]
    )

    print(completion.choices[0].message)

if __name__ == "__main__":
    get_sentiment('You are a BAD person.')
