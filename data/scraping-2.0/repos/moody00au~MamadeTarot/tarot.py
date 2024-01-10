import streamlit as st
import openai
import random
import smtplib
from email.message import EmailMessage

# Use the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Use the email and password from Streamlit Secrets
my_email = st.secrets["gmail"]["my_email"]
my_app_specific_password = st.secrets["gmail"]["my_app_specific_password"]

# Check if the button was previously clicked
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Function to send reading email
def send_reading_email(message, recipient):
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = 'Tarot Reading'
    msg['From'] = st.secrets["gmail"]["my_email"]
    msg['To'] = recipient

    # Establish a connection to the Gmail server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(st.secrets["gmail"]["my_email"], st.secrets["gmail"]["my_app_specific_password"])
    server.send_message(msg)
    server.quit()
    
# Define a dictionary of tarot cards
tarot_deck = [
    'The Fool',
    'The Magician',
    'The High Priestess',
    'The Empress',
    'The Emperor',
    'The Hierophant',
    'The Lovers',
    'The Chariot',
    'Strength',
    'The Hermit',
    'Wheel of Fortune',
    'Justice',
    'The Hanged Man',
    'Death',
    'Temperance',
    'The Devil',
    'The Tower',
    'The Star',
    'The Moon',
    'The Sun',
    'Judgement',
    'The World',
    'Ace of Cups',
    'Two of Cups',
    'Three of Cups',
    'Four of Cups',
    'Five of Cups',
    'Six of Cups',
    'Seven of Cups',
    'Eight of Cups',
    'Nine of Cups',
    'Ten of Cups',
    'Page of Cups',
    'Knight of Cups',
    'Queen of Cups',
    'King of Cups',
    'Ace of Pentacles',
    'Two of Pentacles',
    'Three of Pentacles',
    'Four of Pentacles',
    'Five of Pentacles',
    'Six of Pentacles',
    'Seven of Pentacles',
    'Eight of Pentacles',
    'Nine of Pentacles',
    'Ten of Pentacles',
    'Page of Pentacles',
    'Knight of Pentacles',
    'Queen of Pentacles',
    'King of Pentacles',
    'Ace of Swords',
    'Two of Swords',
    'Three of Swords',
    'Four of Swords',
    'Five of Swords',
    'Six of Swords',
    'Seven of Swords',
    'Eight of Swords',
    'Nine of Swords',
    'Ten of Swords',
    'Page of Swords',
    'Knight of Swords',
    'Queen of Swords',
    'King of Swords',
    'Ace of Wands',
    'Two of Wands',
    'Three of Wands',
    'Four of Wands',
    'Five of Wands',
    'Six of Wands',
    'Seven of Wands',
    'Eight of Wands',
    'Nine of Wands',
    'Ten of Wands',
    'Page of Wands',
    'Knight of Wands',
    'Queen of Wands',
    'King of Wands'
]

celtic_cross_positions = [
    'The Present',
    'The Challenge',
    'The Past',
    'The Future',
    'Above',
    'Below',
    'Advice',
    'External Influences',
    'Hopes and Fears',
    'Outcome'
]

card_to_filename = {
    'Ace of Cups': 'Cups01.jpg',
    'Two of Cups': 'Cups02.jpg',
    'Three of Cups': 'Cups03.jpg',
    'Four of Cups': 'Cups04.jpg',
    'Five of Cups': 'Cups05.jpg',
    'Six of Cups': 'Cups06.jpg',
    'Seven of Cups': 'Cups07.jpg',
    'Eight of Cups': 'Cups08.jpg',
    'Nine of Cups': 'Cups09.jpg',
    'Ten of Cups': 'Cups10.jpg',
    'Page of Cups': 'Cups11.jpg',
    'Knight of Cups': 'Cups12.jpg',
    'Queen of Cups': 'Cups13.jpg',
    'King of Cups': 'Cups14.jpg',
    'Ace of Pentacles': 'Pents01.jpg',
    'Two of Pentacles': 'Pents02.jpg',
    'Three of Pentacles': 'Pents03.jpg',
    'Four of Pentacles': 'Pents04.jpg',
    'Five of Pentacles': 'Pents05.jpg',
    'Six of Pentacles': 'Pents06.jpg',
    'Seven of Pentacles': 'Pents07.jpg',
    'Eight of Pentacles': 'Pents08.jpg',
    'Nine of Pentacles': 'Pents09.jpg',
    'Ten of Pentacles': 'Pents10.jpg',
    'Page of Pentacles': 'Pents11.jpg',
    'Knight of Pentacles': 'Pents12.jpg',
    'Queen of Pentacles': 'Pents13.jpg',
    'King of Pentacles': 'Pents14.jpg',
    'The Fool': 'RWS_Tarot_00_Fool.jpg',
    'The Magician': 'RWS_Tarot_01_Magician.jpg',
    'The High Priestess': 'RWS_Tarot_02_High_Priestess.jpg',
    'The Empress': 'RWS_Tarot_03_Empress.jpg',
    'The Emperor': 'RWS_Tarot_04_Emperor.jpg',
    'The Hierophant': 'RWS_Tarot_05_Hierophant.jpg',
    'The Lovers': 'TheLovers.jpg',
    'The Chariot': 'RWS_Tarot_07_Chariot.jpg',
    'Strength': 'RWS_Tarot_08_Strength.jpg',
    'The Hermit': 'RWS_Tarot_09_Hermit.jpg',
    'Wheel of Fortune': 'RWS_Tarot_10_Wheel_of_Fortune.jpg',
    'Justice': 'RWS_Tarot_11_Justice.jpg',
    'The Hanged Man': 'RWS_Tarot_12_Hanged_Man.jpg',
    'Death': 'RWS_Tarot_13_Death.jpg',
    'Temperance': 'RWS_Tarot_14_Temperance.jpg',
    'The Devil': 'RWS_Tarot_15_Devil.jpg',
    'The Tower': 'RWS_Tarot_16_Tower.jpg',
    'The Star': 'RWS_Tarot_17_Star.jpg',
    'The Moon': 'RWS_Tarot_18_Moon.jpg',
    'The Sun': 'RWS_Tarot_19_Sun.jpg',
    'Judgement': 'RWS_Tarot_20_Judgement.jpg',
    'The World': 'RWS_Tarot_21_World.jpg',
    'Ace of Swords': 'Swords01.jpg',
    'Two of Swords': 'Swords02.jpg',
    'Three of Swords': 'Swords03.jpg',
    'Four of Swords': 'Swords04.jpg',
    'Five of Swords': 'Swords05.jpg',
    'Six of Swords': 'Swords06.jpg',
    'Seven of Swords': 'Swords07.jpg',
    'Eight of Swords': 'Swords08.jpg',
    'Nine of Swords': 'Swords09.jpg',
    'Ten of Swords': 'Swords10.jpg',
    'Page of Swords': 'Swords11.jpg',
    'Knight of Swords': 'Swords12.jpg',
    'Queen of Swords': 'Swords13.jpg',
    'King of Swords': 'Swords14.jpg',
    'Ace of Wands': 'Wands01.jpg',
    'Two of Wands': 'Wands02.jpg',
    'Three of Wands': 'Wands03.jpg',
    'Four of Wands': 'Wands04.jpg',
    'Five of Wands': 'Wands05.jpg',
    'Six of Wands': 'Wands06.jpg',
    'Seven of Wands': 'Wands07.jpg',
    'Eight of Wands': 'Wands08.jpg',
    'Nine of Wands': 'Wands09.jpg',
    'Ten of Wands': 'Wands10.jpg',
    'Page of Wands': 'Wands11.jpg',
    'Knight of Wands': 'Wands12.jpg',
    'Queen of Wands': 'Wands13.jpg',
    'King of Wands': 'Wands14.jpg'
}

base_url = "https://raw.githubusercontent.com/moody00au/MamadeTarot/main/tarot_images/"

# Check if the counter file exists, if not create one
try:
    with open('counter.txt', 'r') as f:
        counter = int(f.read())
except FileNotFoundError:
    with open('counter.txt', 'w') as f:
        f.write('0')
    counter = 0

def get_tarot_reading(spread, question, holistic=False):
    model = "gpt-3.5-turbo"
    
    if holistic:
        spread_description = ". ".join([f"{pos}: {card}" for pos, card in spread.items()])
        prompt_content = f"Given the user's question: '{question}', and the tarot spread: {spread_description}, provide a 3-paragraph holistic interpretation without referring to the cards directly. Instead, focus on the positions and the influences and advice they represent."
    else:
        position, card = list(spread.items())[0]
        prompt_content = f"Given the user's question: '{question}', provide a one-paragraph interpretation of the card {card} in the position {position}, explaining its significance without referring to the card directly. Ensure the reading is beginner-friendly."

    chat_log = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt_content}]
        
    response = openai.ChatCompletion.create(
        model=model,
        messages=chat_log,
    )

    return response['choices'][0]['message']['content'].strip()

st.title('🔮 Tarot Habibi - by Hammoud 🔮')
st.write('Welcome to Tarot Habibi! This app provides tarot card readings using the Celtic Cross spread. Simply enter your question and draw the cards to receive insights into various aspects of your life. If you\'re new to tarot, don\'t worry! Each card\'s meaning will be explained in detail. Ready to begin? Please enter your question below:')

# User enters their question
question = st.text_input('What troubles you my child?', key="user_question")

# Initialize spread as an empty dictionary
spread = {}

# Descriptions for each position
position_descriptions = {
    'The Present': 'Represents your current situation.',
    'The Challenge': 'Indicates the immediate challenge or problem facing you.',
    'The Past': 'Denotes past events that are affecting the current situation.',
    'The Future': 'Predicts the likely outcome if things continue as they are.',
    'Above': 'Represents your goal or best outcome in this situation.',
    'Below': 'Reflects your subconscious influences, fears, and desires.',
    'Advice': 'Offers guidance on how to navigate the current challenges.',
    'External Influences': 'Represents external factors affecting the situation.',
    'Hopes and Fears': 'Indicates your hopes, fears, and expectations.',
    'Outcome': 'Predicts the final outcome of the situation.'
}

deck = tarot_deck.copy()

# User clicks to draw cards for the spread
if st.button('Draw Cards 🃏') and question:
    counter += 1
    with open('counter.txt', 'w') as f:
        f.write(str(counter))
    
    full_reading = ""
    
    # Initialize the deck as a copy of tarot_deck
    deck = tarot_deck.copy()
    
    for position in celtic_cross_positions:
        card = random.choice(deck)
        deck.remove(card)
        
        # Display card name, position, and description in larger, centered, and bold text
        st.markdown(f"<h2 style='text-align: center; font-weight: bold;'>{position}: {card}</h2>", unsafe_allow_html=True)
        
        # Fetch and display the card image with reduced size
        image_url = base_url + card_to_filename[card]
        st.image(image_url, use_column_width='auto', width=300)  # Adjust the width as needed
        
        # Get tarot reading for the drawn card
        reading = get_tarot_reading({position: card}, question)
        full_reading += f"{position}: {card}\n{reading}\n\n"  # Format as needed
        st.write(reading)

    # After generating the Tarot reading and displaying it to the user:
    recipient_email = st.text_input("Enter your email to receive the reading:")
    
    # User clicks to send the reading
    if st.button('Send Reading 📧') and recipient_email:
        email_content = f"Question: {question}\n\n{full_reading}"
        try:
            send_reading_email(email_content, recipient_email)
            st.success("Email sent successfully!")
        except Exception as e:
            st.error(f"Error sending email: {e}")
