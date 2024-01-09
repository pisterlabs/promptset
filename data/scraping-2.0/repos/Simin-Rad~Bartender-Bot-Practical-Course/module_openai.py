# input: customer order converted in text (modul_audio.py) 
# output: specifics of the oders (name of the order, size and the number of drinks)

# Hier we analyse the content of the order-text using openai API and the filter the order 
# specifications from the order-text.

import openai
import sys
import configparser

config = configparser.ConfigParser()
with open('.env', 'r') as f:
    config_string = '[dummy_section]\n' + f.read()

config.read_string(config_string)
key = config['dummy_section']['api_key']
#print (key, file=sys.stderr)

def interface_openai(input_text):
    openai.api_key = key
    #todo define size "default"
    #todo define number"default"
    drink_size_list = "[small, medium, large, double shot]"
    drink_name_list = "[cola, gin, tonic, vodka, Mojito, Margarita, Cosmopolitan, Old Fashioned, Martini, Daiquiri, Piña Colada, Mai Tai, Moscow Mule, Long Island Iced Tea, Negroni, Bloody Mary, Manhattan, White Russian, Caipirinha, Whiskey Sour, Sex on the Beach, Tequila Sunrise, Espresso Martini, Tom Collins, Lager, IPA, Stout, Pilsner, Wheat Beer, Pale Ale, Saison, Porter, Belgian Tripel, Gose, Mojito, Margarita, Cosmopolitan, Old Fashioned, Martini, Daiquiri, Piña Colada, Mai Tai, Moscow Mule, Long Island Iced Tea]"

    prompt = "What drink's name do you see in this sentence and the size and the number of the drinks if mentioned? " + input_text + "print the output in this order and in json format with following keys: name, size, number. name key represents representing the name of the drink. size key represents the size of drink. number key represents number of ordered drinks. size value is one of the following:" + drink_size_list + ". name of the drink is one of the following: " + drink_name_list+". Only send back the json string, nothing else."

    # Make the API call
    completion = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  # Choose an appropriate engine
        prompt=prompt,
        max_tokens=1024  # Adjust as needed
    )

    # Extract the AI's reply from the response
    output_order = completion.choices[0].text.strip()

    output = output_order
    print ("AI:", file=sys.stderr)
    print (output, file=sys.stderr)

    return output

if __name__ == "__main__":
    input_text1 = "I want to order one small glass of Coca-Cola!"
    interface_openai(input_text1)

    input_text2 = "I want to order one small glass of Coca-Cola!"
    interface_openai(input_text2)