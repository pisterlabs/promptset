import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
def sentiment_analysis(text):
    preamble = """

    """

    premise = """
    Classify the following sentence into one of the labels from the following list of emotions: ['Neutral', 'Amazed', 'Foolish', 'Overwhelmed', 'Angry', 'Frustrated', 'Peaceful', 'Annoyed', 'Furious', 'Proud', 'Anxious', 'Grievous', 'Relieved', 'Ashamed', 'Happy', 'Resentful', 'Bitter', 'Hopeful', 'Sad', 'Bored', 'Hurt', 'Satisfied', 'Comfortable', 'Inadequate', 'Scared', 'Confused', 'Insecure', 'Self-conscious', 'Content', 'Inspired', 'Shocked', 'Depressed', 'Irritated', 'Silly', 'Determined', 'Jealous', 'Stupid', 'Disdain', 'Joy', 'Suspicious', 'Disgusted', 'Lonely', 'Tense', 'Eager', 'Lost', 'Terrified', 'Embarrassed', 'Loving', 'Trapped', 'Energetic', 'Miserable', 'Uncomfortable', 'Envious', 'Motivated', 'Worried', 'Excited', 'Nervous', 'Worthless']
    Respond with only one word, which is an emotion from the list. If not sure, respond with only unknown.
    """

    query = """
    Classify the following sentence into an emotion from the list of emotions.
    Respond with only one word, which is an emotion from the list.  If not sure, respond with only unknown.

    """
    while True:
        query = text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "assistant", "content": preamble},
                    {"role": "system", "content": premise},
                    {"role": "user", "content": query}],
            temperature = 0.3,
            max_tokens = 200,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        if len(response['choices'][0]['message']['content'].split(" ")) > 1:
            return "unknown"
        else:
            resp = response['choices'][0]['message']['content']
            return resp.lower().replace('.', '')
