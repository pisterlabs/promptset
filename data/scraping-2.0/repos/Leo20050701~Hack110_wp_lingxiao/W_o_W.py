from openai import OpenAI
from variables import *
import os
import sys
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher


def get_response_from_AI(system_content: str, user_content: str) -> str:
    """Get the response from ChatGPT 4.0."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
        {
            "role": "system", "content": f"{system_content}"
        },
        {
            "role": "user", "content": f"{user_content}"
        }
        ]
    )
    return response.choices[0].message.content


def get_first_bing_image_url(search_query):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://www.bing.com/images/search?q={search_query}"

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        first_image = soup.find('a', class_='iusc')

        if not first_image:
            return 0

        m = first_image.get('m')
        if not m:
            return 0
        
        start = m.find('murl')+7
        end = m.find('"', start)
        image_url = m[start:end]

        return image_url
    
    except Exception as e:
        return 0
    

def compare_strings(str1, str2) -> float:
    """Compare 2 strings."""
    str1 = str1.lower()
    str2 = str2.lower()
    percent_match: float = SequenceMatcher(None, str1, str2)
    return percent_match.ratio() * 100


def main() -> None:
    """Main function."""
    status: bool = True
    win: bool = False
    round: int = 1
    while status:
        print("Movie List.")
        for i in movie_list:
            print(i)
        movie = input("Please select a movie from the list: ")
        character = get_response_from_AI(system_content_select_character, ask_for_character(movie))

        win = False
        round = 1

        while win == False and round <= 4:
            user_guess = ""
            if round == 1:
                print(f"Clue 1: {get_response_from_AI(system_content_clue, first_clue(character, movie))}")
                user_guess = input("Please input your guess: ")
                if compare_strings(user_guess, character) >= 30.0:
                    win = True
                else:
                    round += 1

            if round == 2:
                print(f"Clue 2: {get_response_from_AI(system_content_clue, second_clue(character, movie))}")
                user_guess = input("Please input your guess: ")
                if compare_strings(user_guess, character) >= 30.0:
                    win = True
                else:
                    round += 1

            if round == 3:
                print(f"Clue 3: {get_response_from_AI(system_content_clue, third_clue(character, movie))}")
                user_guess = input("Please input your guess: ")
                if compare_strings(user_guess, character) >= 30.0:
                    win = True
                else:
                    round += 1
            
            if round == 4:
                print(get_first_bing_image_url(f"{character} in {movie}"))
                user_guess = input("Please input your guess: ")
                if compare_strings(user_guess, character) >= 30.0:
                    win = True
                else:
                    round += 1
        
        if win == True:
            print("You Win!!")
        else:
            print("You Loss...")

        print(character)
        print(f"{compare_strings(user_guess, character)}% matched!")

        play_again = input("Do you want to play again? (Y/N)")
        if not (play_again == "Y" or play_again == "y" or play_again == "yes" or play_again == "YES" or play_again == "Yes"):
            status = False