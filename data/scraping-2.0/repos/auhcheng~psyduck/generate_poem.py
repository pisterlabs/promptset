import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

topics = ["the futility of medicine", "passionate love", "natural serenity"]
title = "Wander Away"

def generate_poem(topics, title):
    # Prompt design heavily inspired by https://www.gwern.net/GPT-3#the-universe-is-a-glitch
    prompt = f"""Below is a selection of 10 long-form poems written by the latest cutting-edge contemporary poets. They cover topics ranging from {topics[0]} to {topics[1]} to {topics[2]}, and feature remarkable use of metaphor, rhyme, and meter.
\t"Stopping by Woods on a Snowy Evening"
By Robert Frost
Whose woods these are I think I know.
His house is in the village though;
He will not see me stopping here
To watch his woods fill up with snow.

My little horse must think it queer
To stop without a farmhouse near
Between the woods and frozen lake
The darkest evening of the year.

He gives his harness bells a shake
To ask if there is some mistake.
The only other sound’s the sweep
Of easy wind and downy flake.

The woods are lovely, dark and deep,
But I have promises to keep,
And miles to go before I sleep,
And miles to go before I sleep.

\t"{title}"
By """

    print(repr(prompt))
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.8,
        max_tokens=150,
        top_p=0.85,
        frequency_penalty=0.5,
        presence_penalty=0.25,
        stop=['\t"', '\nBy\n']
    )

    print(response)
    poem = response.choices[0].text
    
    # strip leading/trailing whitespace
    poem = poem.strip()
    first_line = poem[:poem.find('\n')+1]
    if len(first_line.split(' ')) == 2:
        # remove the first line
        poem = poem[poem.find('\n')+1:]
    
    # print(poem)
    return poem

if __name__ == '__main__':
    print(generate_poem(topics, title))