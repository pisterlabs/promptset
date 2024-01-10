import random
from flask import Flask, render_template, request, jsonify
import openai

# Set your OpenAI API key
openai.api_key = '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'

app = Flask(__name__)

tarot_cards = [
    # Major Arcana
    'The Fool', 'The Magician', 'The High Priestess', 'The Empress', 'The Emperor',
    'The Hierophant', 'The Lovers', 'The Chariot', 'Strength', 'The Hermit',
    'Wheel of Fortune', 'Justice', 'The Hanged Man', 'Death', 'Temperance',
    'The Devil', 'The Tower', 'The Star', 'The Moon', 'The Sun',
    'Judgement', 'The World',
    # Minor Arcana
    'Ace of Cups', 'Two of Cups', 'Three of Cups', 'Four of Cups', 'Five of Cups',
    'Six of Cups', 'Seven of Cups', 'Eight of Cups', 'Nine of Cups', 'Ten of Cups',
    'Page of Cups', 'Knight of Cups', 'Queen of Cups', 'King of Cups',
    'Ace of Swords', 'Two of Swords', 'Three of Swords', 'Four of Swords', 'Five of Swords',
    'Six of Swords', 'Seven of Swords', 'Eight of Swords', 'Nine of Swords', 'Ten of Swords',
    'Page of Swords', 'Knight of Swords', 'Queen of Swords', 'King of Swords',
    'Ace of Wands', 'Two of Wands', 'Three of Wands', 'Four of Wands', 'Five of Wands',
    'Six of Wands', 'Seven of Wands', 'Eight of Wands', 'Nine of Wands', 'Ten of Wands',
    'Page of Wands', 'Knight of Wands', 'Queen of Wands', 'King of Wands',
    'Ace of Pentacles', 'Two of Pentacles', 'Three of Pentacles', 'Four of Pentacles', 'Five of Pentacles',
    'Six of Pentacles', 'Seven of Pentacles', 'Eight of Pentacles', 'Nine of Pentacles', 'Ten of Pentacles',
    'Page of Pentacles', 'Knight of Pentacles', 'Queen of Pentacles', 'King of Pentacles'
]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start-reading', methods=['POST'])
def start_reading():
    # Get the question or intention from the request
    question = request.json['question']

    # Shuffle the deck and draw seven cards for the reading
    random.shuffle(tarot_cards)
    drawn_cards = tarot_cards[:7]

    # Use the OpenAI API to generate interpretations for each card
    interpretations = []
    for i, card in enumerate(drawn_cards):
        time_period = [
            'What led to the issue',
            'The current situation',
            'Your emotional state',
            'Your rational side',
            'Aspects of the issue that need acknowledging',
            'Action to take',
            'Advice for the future'
        ][i]
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Interpret the tarot card: {card} in the context of {time_period} for the question: {question}",
            max_tokens=100
        )
        interpretations.append(response.choices[0].text.strip())

    # Use the OpenAI API to generate a holistic understanding of the reading
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Given the cards drawn ({', '.join(drawn_cards)}) and their interpretations ({', '.join(interpretations)}), provide a holistic understanding of the tarot reading for the question: {question}",
        max_tokens=100
    )
    holistic_understanding = response.choices[0].text.strip()

    # Return the drawn cards, their interpretations, and the holistic understanding
    return jsonify({'cards': drawn_cards, 'interpretations': interpretations, 'holistic_understanding': holistic_understanding})

if __name__ == '__main__':
    app.run(debug=True)
