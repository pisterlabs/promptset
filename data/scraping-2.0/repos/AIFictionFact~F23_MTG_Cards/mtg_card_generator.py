import openai

# Set API key
openai.api_key = 'put openai API key here'

def generate_magic_card_name(card_features):
    # Define a prompt
    prompt = f"Create only the name of a new Magic: The Gathering card that has the following attributes:\n\nMana costs: "

    # card features is a list of 12 details for writing the card
    # 0 to 5 are mana colors and amounts. Red,blue,green,white,black,colorless
    if card_features[0] > 0:     # if there is red mana
        prompt += f"\n{card_features[0]} Red"
    if card_features[1] > 0:     # if there is blue mana
        prompt += f"\n{card_features[1]} Blue"
    if card_features[2] > 0:     # if there is green mana
        prompt += f"\n{card_features[2]} Green"
    if card_features[3] > 0:     # if there is white mana
        prompt += f"\n{card_features[3]} White"
    if card_features[4] > 0:     # if there is black mana
        prompt += f"\n{card_features[4]} Black"
    if card_features[5] > 0:     # if there is colorless mana
        prompt += f"\n{card_features[5]} Colorless"

    # check if no mana cost
    if card_features[0] + card_features[1] + card_features[2] + card_features[3] + card_features[4] + card_features[5] == 0:
        prompt += f"\n(no mana cost)"

    prompt += f"\n\nCard Types: "
    # 6 to 10 are card types creature, instant, sorcery, artifact, enchantment
    if card_features[6] == 1:
        prompt += f"\n Creature"
    if card_features[7] == 1:
        prompt += f"\n Instant"    
    if card_features[8] == 1:
        prompt += f"\n Sorcery"    
    if card_features[9] == 1:
        prompt += f"\n Artifact"    
    if card_features[10] == 1:
        prompt += f"\n Enchantment"

    prompt += f"\n\nPurpose: "
    # 11 is a special feature to indicate the level of specialization the card should have
    # 0-0.4 you get a general card 
    # 0.4-0.6 you get a card good for the color
    # 0.6-0.8 you get a commander type card
    # 0.8-1 you get a infinite combo card (hopefully)
    if card_features[11] < 0.4:
        prompt += f"\n All-rounder card for any deck"
    elif card_features[11] < 0.6:
        prompt += f"\n Card which is good for decks of the same color combination"    
    elif card_features[11] < 0.8:
        prompt += f"\n Impactful card you have to build a deck around to be good"    
    elif card_features[11] <= 1:
        prompt += f"\n Special card which is extremely powerful if paired with other abilities"

    prompt += f"\n\nName:"

    # Generate card text using the GPT-3 model
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,  # Adjust card text length
        temperature=0.7,  # Adjust randomness
        n=1  # Number of cards
    )

    # Extract card text
    card_text = response.choices[0].text.strip()

    # Print
    # print(card_text)
    return card_text


def generate_magic_card(card_name, card_features):
    # Define a prompt

    prompt = f'Create the abilities of a new Magic: The Gathering card, "{card_name}" that has the following attributes:\n\nMana costs: '

    # card features is a list of 12 details for writing the card
    # 0 to 5 are mana colors and amounts. Red,blue,green,white,black,colorless
    mana_info = ""

    if card_features[0] > 0:     # if there is red mana
        mana_info += f"\n{card_features[0]} Red"
    if card_features[1] > 0:     # if there is blue mana
        mana_info += f"\n{card_features[1]} Blue"
    if card_features[2] > 0:     # if there is green mana
        mana_info += f"\n{card_features[2]} Green"
    if card_features[3] > 0:     # if there is white mana
        mana_info += f"\n{card_features[3]} White"
    if card_features[4] > 0:     # if there is black mana
        mana_info += f"\n{card_features[4]} Black"
    if card_features[5] > 0:     # if there is colorless mana
        mana_info += f"\n{card_features[5]} Colorless"

    # check if no mana cost
    if card_features[0] + card_features[1] + card_features[2] + card_features[3] + card_features[4] + card_features[5] == 0:
        mana_info += f"\n(no mana cost)"


    prompt += mana_info
    mana_info += "\n"

    prompt += f"\n\nCard Types: "
    # 6 to 10 are card types creature, instant, sorcery, artifact, enchantment
    type_info = ""

    if card_features[6] == 1:
        type_info += f"\n Creature"
    if card_features[7] == 1:
        type_info += f"\n Instant"    
    if card_features[8] == 1:
        type_info += f"\n Sorcery"    
    if card_features[9] == 1:
        type_info += f"\n Artifact"    
    if card_features[10] == 1:
        type_info += f"\n Enchantment"

    prompt += type_info

    prompt += f"\n\nPurpose: "
    # 11 is a special feature to indicate the level of specialization the card should have
    # 0-0.4 you get a general card 
    # 0.4-0.6 you get a card good for the color
    # 0.6-0.8 you get a commander type card
    # 0.8-1 you get a infinite combo card (hopefully)
    if card_features[11] < 0.4:
        prompt += f"\n All-rounder card for any deck"
    elif card_features[11] < 0.6:
        prompt += f"\n Card which is good for decks of the same color combination"    
    elif card_features[11] < 0.8:
        prompt += f"\n Impactful card you have to build a deck around to be good"    
    elif card_features[11] <= 1:
        prompt += f"\n Special card which is extremely powerful if paired with other abilities"

    prompt += f"\n\nCard Text:"

    # Generate card text using the GPT-3 model
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,  # Adjust card text length
        temperature=0.7,  # Adjust randomness
        n=1  # Number of cards
    )

    # Extract card text
    card_text = response.choices[0].text.strip()

    return(card_text)


def generate_card_art(card_name):

    # Construct a prompt for DALL·E
    desc = f"Create a detailed epic fantasy oil painting to act as art for a Magic: The Gathering card named '{card_name}'."

    response = openai.Image.create(
        model="dall-e-2",
        prompt=desc,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url

    return image_url


inappropriate_words = ["White Power", "White Knight", "arse", "bitch", "bullshit", "piss", "bellend", "shit" "bollocks", "fuck", "prick", "bastard", "cock", "Black Sun's Zenith"]
# search a card's name and description for inappropriate words and phrases.
def contains_inappropriate_words(text):
    text_lower = text.lower()
    for word in inappropriate_words:
        if word in text_lower:
            return True
    return False

# Exaple usage
# I want to make a 3 cost blue/colorless spell card meant for an infinite combo
'''
features = [0,2,0,0,0,1,0,0,1,0,0,1]

name = generate_magic_card_name(features)
card_details = generate_magic_card(name, features)
card_art = generate_card_art(name)
'''
