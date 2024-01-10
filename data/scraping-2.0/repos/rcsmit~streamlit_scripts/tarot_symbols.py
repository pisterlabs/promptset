import openai

from keys_dummy import *
import random
import pandas as pd
    
import streamlit as st
from PIL import Image
# Set up OpenAI API credentials
openai.api_key = OPENAI_API_KEY


filenames = ["RWS_Tarot_00_Fool.jpg",
                "RWS_Tarot_01_Magician.jpg",
                "RWS_Tarot_02_High_Priestess.jpg",
                "RWS_Tarot_03_Empress.jpg",
                "RWS_Tarot_04_Emperor.jpg",
                "RWS_Tarot_05_Hierophant.jpg",
                "RWS_Tarot_06_The_Lovers.jpg",
                "RWS_Tarot_07_Chariot.jpg",
                "RWS_Tarot_08_Strength.jpg",
                "RWS_Tarot_09_Hermit.jpg",
                "RWS_Tarot_10_Wheel_of_Fortune.jpg",
                "RWS_Tarot_11_Justice.jpg",
                "RWS_Tarot_12_Hanged_Man.jpg",
                "RWS_Tarot_13_Death.jpg",
                "RWS_Tarot_14_Temperance.jpg",
                "RWS_Tarot_15_Devil.jpg",
                "RWS_Tarot_16_Tower.jpg",
                "RWS_Tarot_17_Star.jpg",
                "RWS_Tarot_18_Moon.jpg",
                "RWS_Tarot_19_Sun.jpg",
                "RWS_Tarot_20_Judgement.jpg",
                "RWS_Tarot_21_World.jpg",
                "Wands01.jpg",
                "Wands02.jpg",
                "Wands03.jpg",
                "Wands04.jpg",
                "Wands05.jpg",
                "Wands06.jpg",
                "Wands07.jpg",
                "Wands08.jpg",
                "Wands09.jpg",
                "Wands10.jpg",
                "Wands11.jpg",
                "Wands12.jpg",
                "Wands13.jpg",
                "Wands14.jpg",
                "Cups01.jpg",
                "Cups02.jpg",
                "Cups03.jpg",
                "Cups04.jpg",
                "Cups05.jpg",
                "Cups06.jpg",
                "Cups07.jpg",
                "Cups08.jpg",
                "Cups09.jpg",
                "Cups10.jpg",
                "Cups11.jpg",
                "Cups12.jpg",
                "Cups13.jpg",
                "Cups14.jpg",
                "Swords01.jpg",
                "Swords02.jpg",
                "Swords03.jpg",
                "Swords04.jpg",
                "Swords05.jpg",
                "Swords06.jpg",
                "Swords07.jpg",
                "Swords08.jpg",
                "Swords09.jpg",
                "Swords10.jpg",
                "Swords11.jpg",
                "Swords12.jpg",
                "Swords13.jpg",
                "Swords14.jpg",
                "Pents01.jpg",
                "Pents02.jpg",
                "Pents03.jpg",
                "Pents04.jpg",
                "Pents05.jpg",
                "Pents06.jpg",
                "Pents07.jpg",
                "Pents08.jpg",
                "Pents09.jpg",
                "Pents10.jpg",
                "Pents11.jpg",
                "Pents12.jpg",
                "Pents13.jpg",
                "Pents14.jpg"]

cards = ["The Fool",
        "The Magician",
        "The High Priestess",
        "The Empress",
        "The Emperor",
        "The Hierophant",
        "The Lovers",
        "The Chariot",
        "Strength",
        "The Hermit",
        "Wheel of Fortune",
        "Justice",
        "The Hanged Man",
        "Death",
        "Temperance",
        "The Devil",
        "The Tower",
        "The Star",
        "The Moon",
        "The Sun",
        "Judgment",
        "The World",
        "Ace of Wands",
        "Two of Wands",
        "Three of Wands",
        "Four of Wands",
        "Five of Wands",
        "Six of Wands",
        "Seven of Wands",
        "Eight of Wands",
        "Nine of Wands",
        "Ten of Wands",
        "Page of Wands",
        "Knight of Wands",
        "Queen of Wands",
        "King of Wands",
        "Ace of Cups",
        "Two of Cups",
        "Three of Cups",
        "Four of Cups",
        "Five of Cups",
        "Six of Cups",
        "Seven of Cups",
        "Eight of Cups",
        "Nine of Cups",
        "Ten of Cups",
        "Page of Cups",
        "Knight of Cups",
        "Queen of Cups",
        "King of Cups",
        "Ace of Swords",
        "Two of Swords",
        "Three of Swords",
        "Four of Swords",
        "Five of Swords",
        "Six of Swords",
        "Seven of Swords",
        "Eight of Swords",
        "Nine of Swords",
        "Ten of Swords",
        "Page of Swords",
        "Knight of Swords",
        "Queen of Swords",
        "King of Swords",
        "Ace of Pentacles",
        "Two of Pentacles",
        "Three of Pentacles",
        "Four of Pentacles",
        "Five of Pentacles",
        "Six of Pentacles",
        "Seven of Pentacles",
        "Eight of Pentacles",
        "Nine of Pentacles",
        "Ten of Pentacles",
        "Page of Pentacles",
        "Knight of Pentacles",
        "Queen of Pentacles",
        "King of Pentacles"]

def read_tarot_cards(cards):
    number_of_cards = 3
    # Initialize an empty prompt string
    prompt = "I have a Tarot reading. My cards are: "

    # Select random cards and add them to the prompt string
    for i in range(number_of_cards):
        # Generate a random number between 1 and the length of the cards list
        random_number = random.randint(0, len(cards)-1)
        
        # Select a card based on the random number generated
        card = cards[random_number]
        
        # Append the card to the prompt string
        if i == number_of_cards - 1:
            prompt += " and " + card
        elif i == 0:
            prompt += " " + card
        else:
            prompt += ", " + card
    prompt += ". Please provide a very short reading."
    # Print the prompt string
    print(prompt)

    answer = ask_gpt(prompt)
    print (answer)


def ask_gpt(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ])
    
    # print (completion)
    if completion.choices[0].message.content!=None:
        return (completion.choices[0].message.content)
    

#cards = ["The Fool", "Two of Wands"]
def do_reading():
    st.header ("Read Tarot Cards")
    if st.button("Read"):
        read_tarot_cards(cards)

def give_symbolic(cards):

    """Search with Chat-GPT for the meanings of the symbols on the cards
    """    
    for i,c in enumerate(cards):

        prompt = f"Please give the symbolic elements found on the {c} tarot card in a tabular format. In the table there are three columnns. The first column is the name of the card, the second the symbolic element and the third the symbolic meaning. Seperate the columns with a '#'"
        answer = ask_gpt(prompt)
        print (f"{i+1}. {c}")
        print (answer)
        with open('tarotmeanings.txt', mode='a') as file:
            file.write(f"{c}\n")
            file.write(answer)
            file.write("\n")

def show_symbols(first_choice, second_choice, cards):
    sheet_id = "1c-YXs7hAvheN4zvuV99lVaIlrQ_g8VDwmLb-Lq8JWDo"
    sheet_name = "symbols"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    df_symbols_ = pd.read_csv(url, delimiter=",")
    df_symbols = df_symbols_.fillna('_')
    # df_symbols = df_symbols_.groupby(['Name of Card','Symbolic Element'])['Symbolic Meaning'].agg(' '.join).reset_index()
   
    if first_choice =="All":
        for card in cards:
            df_card = df_symbols[df_symbols["Name of Card"] == card].groupby(['Name of Card','Symbolic Element'])['Symbolic Meaning'].agg(' / '.join).reset_index()
   
            show_symbols_for_specific_card(card,  df_card)
    elif first_choice == "Major":
        card = second_choice
        
        df_card = df_symbols[df_symbols["Name of Card"] == card].groupby(['Name of Card','Symbolic Element'])['Symbolic Meaning'].agg(' / '.join).reset_index()
   
        show_symbols_for_specific_card(card,  df_card)
    elif first_choice == "Symbols":
        df_card = df_symbols.groupby(['Symbolic Element'])['Symbolic Meaning'].agg(' / '.join).reset_index()
        st.table(df_card)
   
    else:
        card = f"{second_choice} of {first_choice}"
        df_card = df_symbols[(df_symbols["Suit"] == first_choice) & (df_symbols["Number_txt"] == second_choice)].groupby(['Name of Card','Symbolic Element'])['Symbolic Meaning'].agg(' / '.join).reset_index()
   
        show_symbols_for_specific_card(card, df_card)


def show_symbols_for_specific_card(card,df_card):

    filename  = filenames[cards.index(card)]
    #path = "C://Users//rcxsm//Downloads//Tarot carfds//"
    path = "img_tarot_cards/"
   
    st.subheader(card)
            # Open the image file using PIL
    image = Image.open(path + filename)

    image_height = 269  # Replace with your desired image height in pixels

            # Open the image file using PIL
        
            # Resize the image to the desired height
    aspect_ratio = image.width / image.height
    resized_image = image.resize((int(image_height * aspect_ratio), image_height))

    col1, col2 = st.columns(2)
        
    with col1:
        col1.width = 0.2

        # Display the resized image
        st.image(resized_image, caption=card, use_column_width=False)  
    with col2:

        col1.width = 0.8
                # Display an empty spacer to reserve space for the image   
        df_card = df_card[["Name of Card","Symbolic Element","Symbolic Meaning"]]
    
        st.table (df_card)


def main():

    st.header("Tarot card symbols")
    #give_symbolic(cards)
    first_choice = st.sidebar.selectbox("Choose All/Major/Suit/Symbols", ["All", "Major", "Cups", "Swords", "Wands", "Pentacles", "Symbols"], 1)
    if first_choice == "All":
        second_choice = None
    elif first_choice == "Major":
        major_arcana = cards[:22]
        second_choice = st.sidebar.selectbox ("Choose a card", major_arcana, index = 0)
    elif first_choice == "Symbols":
        second_choice = None
    
    else:
        minor_arcana_numbers = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Page", "Knight", "Queen", "King"]
        second_choice = st.sidebar.selectbox ("Second choice", minor_arcana_numbers, index = 0)

    show_symbols(first_choice, second_choice, cards)

    st.info("The symbols and the meanings are retrieved by Chat-GPT. The immages is the actual 1909 deck, no longer under copyright. This image scanned by Holly Voley and retrieved from https://www.sacred-texts.com/tarot by Wikipedia-contributers")

if __name__=='__main__':
    main()