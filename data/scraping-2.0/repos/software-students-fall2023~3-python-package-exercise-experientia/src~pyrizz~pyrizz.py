import os
import random
import pathlib
import openai

# Uncomment when using pytest and uploading the package to PyPi
from pyrizz.pickuplines import pickuplines
from pyrizz.templates import templates

# Uncomment when testing the __main__.py file locally
# from pickuplines import pickuplines
# from templates import templates

PROJECT_ROOT = f"{pathlib.Path(__file__).parent.resolve()}/"

def get_lines(category='all'): 
    if category not in pickuplines:
        return ("Category does not exist!")

    else: 
        return pickuplines[category]

def get_random_line(): 
    allpickuplines = get_lines('all')
    return random.choice(allpickuplines)

def get_random_category_line(category='all'):
    category_pickupline = get_lines(category)
    return random.choice(category_pickupline)

def init_openai(key):
    try:
        openai.api_key = key
        openai.Model.list()
        print("\nSuccessful. You can use AI functionality now.\n")
        return openai
    except openai.error.AuthenticationError as err:
        print("\nAuthentication failed. Try entering a different API key or call on pyrizz.init_openai(your_key).\n")
        return err

def get_ai_line(category, client) -> str:
    try:
        if (category != "" and len(category) <= 50):
            response = client.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages =
                    [{"role": "user", "content": f"I need a {category} pick-up line."},]
            )

            message = response.choices[0]['message']
            ai_line = "{}".format(message['content'])
            return ai_line
        
        elif (category != "" and len(category) > 50):
            return "Please specify a category that is less than 50 characters."
        
        else:
            return "Please specify a category."
            
    except Exception as err:
        return str(err)

def rate_line(pickup_line, client) -> str:
    try:
        if (pickup_line != ""):
            response = client.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages =
                    [{"role": "user", "content": f"Rate this pickup line out of 10 (whole numbers only): {pickup_line} In your response, STRICTLY follow the format of (nothing else): rating/10 - snazzy comment."},]
            )
            message = response.choices[0]['message']
            ai_rating_response = "{}".format(message['content'])
            return ai_rating_response
        
        else:
            return "No pickup line? You gotta use our other features before you come here buddy."
        
    except Exception as err:
        return str(err)

def load_ascii_art(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    ascii_art_pieces = content.split('[End]')
    cleaned_ascii_art_pieces = []
    for piece in ascii_art_pieces:
        if '[Start]' in piece:
            start_index = piece.find('[Start]') + len('[Start]')
            art_piece = piece[start_index:]
            cleaned_ascii_art_pieces.append(art_piece)
    return cleaned_ascii_art_pieces

def create_line(template_number, words):
    if not (0 <= template_number < len(templates)):
        return None, "Template number out of range. Please choose between 0 and {}.".format(len(templates) - 1)

    template_to_use = templates[template_number]
    placeholders_count = template_to_use.count("{}")
    if placeholders_count != len(words):
        return None, "Incorrect number of words provided for the placeholders. Expected {}, got {}.".format(placeholders_count, len(words))

    try:
        user_line = template_to_use.format(*words)
    except IndexError as e:
        return None, f"Error in formatting: {str(e)}"

    if is_line_valid(user_line):
        ascii_art_pieces = load_ascii_art(PROJECT_ROOT + 'data/ascii_art.txt')
        art = random.choice(ascii_art_pieces)
        user_line_with_art = f"{art}\n\n{user_line}"
        return user_line_with_art
    else:
        return None, "Your line doesn't pass our checks. Sorry!"

def get_user_input_for_line():
    print("Choose a template number (0-{}):".format(len(templates) - 1))
    try:
        template_number = int(input("> "))  
    except ValueError:
        print("Invalid input. Please enter an integer value.")
        return None, None

    if template_number not in range(len(templates)):
        print("Invalid template number. Please choose a number between 0 and {}.".format(len(templates) - 1))
        return None, None

    template_to_show = templates[template_number]
    placeholders_count = template_to_show.count("{}")

    print("Fill in the blanks for the following template:")
    print(template_to_show.format(*(['______'] * placeholders_count)))

    print(f"Enter {placeholders_count} word(s) separated by commas to fill into the template:")
    words_input = input("> ")
    words = [word.strip() for word in words_input.split(',')]

    if len(words) != placeholders_count:
        print(f"Incorrect number of words. You need to provide exactly {placeholders_count} word(s).")
        return None, None

    return template_number, words

def is_line_valid(user_line):
    if len(user_line) > 140:
        print("Your pick-up line is too long.")
        return False
    
    return True

def list_templates():
    return templates
