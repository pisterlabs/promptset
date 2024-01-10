import cachetools
import openai
# from dotenv import load_dotenv

# Initialize OpenAI API
openai.api_key = "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"


# Create a cache with a maximum size of 1000 items
cache = cachetools.LRUCache(maxsize=1000)

def get_prediction(text):
    # Check if the prediction is in the cache
    if text in cache:
        print("Cache hit!")
        return cache[text]
    
    # If not in cache, get prediction from GPT-3
    print("Cache miss! Fetching from GPT-3...")
    response = openai.Completion.create(engine="text-davinci-003", prompt=text, max_tokens=60)
    
    # Store the prediction in the cache
    cache[text] = response.choices[0].text.strip()
    
    return cache[text]


if __name__=="__main__":

    print(get_prediction(text="What is the Capital of France? "))
    print(get_prediction(text="What is the Capital of France? "))