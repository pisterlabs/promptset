import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

example = "I am very happy with the service provided by the company."

# not working and I don't know why .. maybe this doesn't actually work... as in davinci can't do it..

def get_sentiment(text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Sentiment analysis of the following text:\n{text}\n",
        temperature=0.5,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )

    #print(response)
    sentiment = response.choices[0].text.strip()
    return sentiment

if __name__ == "__main__":
    text = "crashing bombs and exploding cars and wars and dying people"
    sentiment = get_sentiment(text)
    print(f"The sentiment of the text is {sentiment}")