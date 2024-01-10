from openai import OpenAI
from transformers import pipeline
import pdb

def query_chatgpt(prompt):
    client = OpenAI(
        api_key="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"
    )
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt},
    ]
    )
    return completion.choices[0].message

def format_message(message):
    lines = message.split('\n')
    characters = lines[-1]
    parts = characters.split(':')
    formatted_script = []
    if len(parts) == 2:
        characters = [name.strip() for name in parts[1].split(',')]
        
    for l in lines[:-1]:
        parts = l.split(':', 1)
        if len(parts) == 2:
            character, line = parts[0].strip(), parts[1].strip()
            formatted_script.append((character, line))
    return formatted_script, characters

def filter_lines(script, character, characters):
    
    char_script = []
    for l in script:
        part, line = l[0], l[1]
        if part == character:
            char_script.append(line)
    return char_script

def sentiment_classification(script):
    
    classifier = pipeline("text-classification", model="bdotloh/distilbert-base-uncased-empathetic-dialogues-context",top_k=3)
    script_with_sentiment = []
    for l in script:
        sentiment = classifier(l)
        sentiment_list = []
        for i in range(len(sentiment[0])):
            sentiment_list.append(sentiment[0][i]['label'])
        script_with_sentiment.append((l, sentiment_list))
    return script_with_sentiment


def main():
    prompt = '''
            This is a script from a play. Convert this into a cleanly formatted version.  Remove stage directions from it which are marked in paranthesis.
            At the end output all the characters in the play in the form Characters: Character1, Character2, ...

            Do not show anyway that you are a generative AI or chatbot of any kind.
            '''

    with open('speech.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    #result = query_chatgpt(prompt + content)
    #message = result.content
    message = """
    EDDIE: Luke and I are inseparable. Well, we used to be.\nROXANNE: Whoa! Careful with the sprinkles!\nLUKE: Oh, sorry.\nEDDIE: Did you hear that? Careful with the sprinkles? She's so controlling!\nLUKE: How's it look so far?\nROXANNE: More frosting on this side.\nEDDIE: And demanding! Luke never used to worry about where he sprinkled...awkward. I hate myself. And he definitely wouldn't let some girl tell him how to make a cake. Who makes cakes together? You know, if a miracle ever happens and I ever end up in Wendy's pan—with Wendy, then she would want to hang out with my other friends! And by other friends, I mean Luke. She's that kind of wonderful girl. All Roxanne ever wants to do is hang out with Luke and nobody else. She probably doesn't care about me at all.\nROXANNE: So how's Eddie doing?\nEDDIE: All right—that's not fair.\nLUKE: I don't know. Haven't seen much of him lately. He always seems too preoccupied with Wendy.\nEDDIE: Is he serious? Me preoccupied with Wendy? When was the last time he invited me to the movies with him and Roxanne? Yeah, also awkward. I'm the third wheel. I'm going to live the rest of my life as the third wheel. I'll never find my own wheel. I'll never be a bicycle. I'll always be the big, honking wheel on some little girl's tricycle. That's me. Eddie the third wheel.\nROXANNE: He likes Wendy?\nEDDIE: He better not—\nLUKE: Yeah. Where have you been?\nEDDIE: I'm gonna kill 'em.\nROXANNE: That's never going to happen.\nEDDIE: Excuse me?\nLUKE: Why not? Eddie's a great guy.\nEDDIE: This is why we're friends.\nROXANNE: Eddie isn't the kind of guy Wendy's interested in.\nLUKE: Why not?\nEDDIE: Yeah, why not?\nROXANNE: He's too nice.\nLUKE: What?\nEDDIE: Typical.\nROXANNE: Plus, he's really awkward.\nEDDIE: I don't know where she gets that from.\nROXANNE: And they're friends. It would be like dating her brother.\nEDDIE: Why does everyone say that!? I'm not friends with my brother!\nLUKE: We were friends before we dated.\nROXANNE: Yeah, but...you're not Eddie.\nEDDIE: What is that supposed to mean?\nLUKE: Oops.\nROXANNE: Oh, it's okay. Don't worry about it.\nLUKE: Hey! This tastes pretty good.\nROXANNE: That's because I made it.\nEDDIE: This is disgusting.\nLUKE: I love you.\nROXANNE: Do you really?\nLUKE: Why don't you ever believe me?\nROXANNE: Prove it.\nLUKE: Why do I have to prove it? Isn't it enough proof that I'm here?\nROXANNE: How is that proof? Where else would you be?\nLUKE: I don't know... Nowhere.\nEDDIE: Nowhere? Nowhere? What about with me?\nROXANNE: I knew it.\nEDDIE: That's it! I'm done.\nROXANNE: I hope he likes it.\nEDDIE: I hope they have a great time together.\nLUKE: Eddie!\nROXANNE: We've made—\nEDDIE: (Takes the cake and shoves it in both their faces. He storms off.) —a cake for your birthday.\n(Blackout.)\n\nCharacters: Eddie, Luke, Roxanne"""
    formatted_script, characters = format_message(message)
    filtered_script = filter_lines(formatted_script, 'EDDIE', characters)
    annotated_script = sentiment_classification(filtered_script)
    print(annotated_script)
    return annotated_script

if __name__ == "__main__":
    main()