import openai
import re

api_key = os.getenv('OPENAI_KEY')
openai.api_key = api_key

def get_sentiment(text):

    prompt_text = """Classify the sentiment of the following tweet as positive, negative, or neutral towards 'company name'. 
    text: {}
    sentiment: """.format(text)

    sentiment = openai.Completion.create(
                  model="text-davinci-003",
                  prompt = prompt_text,
                  max_tokens= 15,
                  temperature=0,
                  )

    # remove special characters e.g n etc, from response
    sentiment = re.sub('W+','', sentiment['choices'][0]['text'])
    
    return sentiment

