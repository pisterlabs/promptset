
from os import environ, path
from tqdm import tqdm
import random

import openai
import csv

NUM_DATAPOINTS = 10
DATA_DIR='data_dir'

# RED, YELLOW, GREEN, BLUE

openai.api_key = environ["OPENAI_APIKEY"]

# Function to send a message to the OpenAI chatbot model and return its response

def get_response(history, character, room, is_imposter):
    
    player_char = "the Imposter" if is_imposter else "not the Imposter"
    
    # EXTRA STRINGS: "Utilize the following format for response: [next_player, response_string]. The value of 'next_player' is the player that you are talking to, and 'response_string' is your response."
    
    message_log = [
        {"role": "system", "content": f"You are a twitch streamer playing a casual game of 'Among Us' with your friends. Use Among Us slang very liberally. There are four characters in this game: Blue, Red, Green, Yellow. You are currently the player {character} and are {player_char}. The room that you are currently located in is {room}. Respond to the given prompts the way that your player would respond."}
    ]
    
    message_log.append({"role": "user", "content": f"The following tab-delineated history of the current emergency meeting conversation: '{history}'. What is your response to the given conversation? Put your response in quotes."})

    # Add a message from the chatbot to the conversation history
    message_log.append(
        {"role": "assistant", "content": "You are a helpful assistant."})
    
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        # The conversation history up to this point, as a list of dictionaries
        messages=message_log,
        # The maximum number of tokens (words or subwords) in the generated response
        max_tokens=1000,
        # The stopping sequence for the generated response, if any (not used here)
        stop=None,
        # The "creativity" of the generated response (higher temperature = more creative)
        temperature=1,
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    return response.choices[0].message.content

def get_initial_message(cur_player, is_imposter, room_seen, dead_player, defendant=None):
    player_char = "the Imposter, and do not want anyone to know" if is_imposter else "not the Imposter"
    suspect_script = f' and are suspicious of {defendant}' if defendant else ''
     
    message_log = [
        {"role": "system", "content": f"You are a twitch streamer playing a casual game of 'Among Us' with your friends. Use Among Us slang very liberally. There are four characters in this game: Blue, Red, Green, Yellow. You are currently the player {cur_player} and are {player_char}. The room that you are currently located in is {room_seen}. You have just called a meeting because you have found {dead_player} to be dead{suspect_script}. Respond to the given prompts the way that your player would respond."}
    ]
    
    message_log.append({"role": "user", "content": "Explain to the other players why you have called a meeting. Put this response in quotation."})

    # Add a message from the chatbot to the conversation history
    message_log.append(
        {"role": "assistant", "content": "You are a helpful assistant."})
    
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        # The conversation history up to this point, as a list of dictionaries
        messages=message_log,
        # The maximum number of tokens (words or subwords) in the generated response
        max_tokens=1000,
        # The stopping sequence for the generated response, if any (not used here)
        stop=None,
        # The "creativity" of the generated response (higher temperature = more creative)
        temperature=1,
    )

    # Find the first response from the chatbot that has text in it (some responses may not have text)
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    return response.choices[0].message.content
        
def generate_vote_script(who_to_vote_off, who_starts_vote):
    ar = ["Red", "Yellow", "Green", "Blue"]
    starting_point_index = ar.index(who_starts_vote)
    ar = ar[starting_point_index:] + ar[:starting_point_index]
    
    ar_without_imposter = ar.copy()
    ar_without_imposter.remove(who_to_vote_off)
    
    ret_text = ''
    for person in ar:
        if person == who_to_vote_off:
            ret_text = ret_text + f'{person}:{ar_without_imposter[(random.randint(0, 2))]}|'
        else:
            ret_text = ret_text + f'{person}:{who_to_vote_off}|'
        
    return ret_text[:-1]



def run_one_training_round(cur_imposter, starting_speaker, who_is_dead, remaining_players):
    
    room_possibilities = ['Upper Engine', 'MedBay', 'Reactor', 'Security', 
                          'Electrical', 'Lower Engine', 'Storage', 'Admin',
                          'O2', 'Shields', 'Navigation', 'Weapons', 
                          'Cafeteria', 'Communications', 'Cargo Bay', 'Cockpit']
    
    speaker_in_same_room = bool(random.randint(0,1))
    
    # Compute 
    remaining_players_minus_imposter = remaining_players.copy()
    remaining_players_minus_imposter.remove(cur_imposter)
    
    #
    starting_point_index = remaining_players.index(starting_speaker)
    impostor_index = remaining_players.index(cur_imposter)
    
    # 
    remaining_players = remaining_players[starting_point_index:] + remaining_players[:starting_point_index]
    
    room_list = [room_possibilities[random.randint(0, len(room_possibilities) - 1)] for _ in remaining_players]
    
    if speaker_in_same_room:
        room_list[starting_point_index] = room_list[impostor_index]
        defendant = cur_imposter
    else:
        defendant = None
        
    initial_response = get_initial_message(starting_speaker, False, room_list[remaining_players.index(cur_imposter)], who_is_dead, defendant)
    history = [(starting_speaker, initial_response)]
    
    ret_history = f'{starting_speaker}:{initial_response}'
    
    for i, player in enumerate((3 * remaining_players)[1:]):
        new_response = get_response('\t'.join(list(map(lambda x : f'{x[0]}:{x[1]}', history))), player, room_list[i % len(room_list)], ((i%len(room_list)) == impostor_index))
        ret_history = ret_history + f'\t{player}:{new_response}'
        
        if len(history) < 4:
            history.append((player, new_response))
        else:
            history = history[1:] + [(player, new_response)]
    
    return ret_history

# def gen_convo_datapoints(n, data_dir='data'):
#     if not isdir(data_dir):
#         mkdir(data_dir)
        
#     with open(join(data_dir, 'gen'), 'w') as file:
#         for i in tqdm(range(n)):
            

# Main function that runs the chatbot
def main():
    
    for convo_number in range(NUM_DATAPOINTS):

        # Define the set of total players

        available_players = ['Red', 'Green', 'Yellow', 'Blue']

        # Randomly sample without replacement from the available players

        # Assign a player to be the dead player initiating the conversation
        who_is_dead = random.choice(available_players)
        available_players.remove(who_is_dead)

        # Assign a character to be an imposter(one max imposter per game)
        cur_imposter = random.choice(available_players)
        available_players.remove(cur_imposter)

        # Assign a player to begin conversation in chat
        starting_speaker = random.choice(available_players)
        available_players.remove(starting_speaker)


        available_players.append(cur_imposter)
        available_players.append(starting_speaker)

        living_players = available_players

        ret_str = run_one_training_round(cur_imposter, starting_speaker, who_is_dead, living_players)
        
        player_responding = []
        player_response = []
        
        if 'As an AI language model' not in ret_str:
            for response in ret_str.split('\t'):
                player_responding.append(response[:response.index(':')])
                player_response.append(response[response.index(':') + 1:])
            
            with open(path.join(DATA_DIR, f'amongusdata{convo_number}.csv'), 'w', newline='') as file:
                for player, response in zip(player_responding, player_response):
                    file.write(player + '\t' + response + '\n')
    
# Call the main function if this file is executed directly (not imported as a module)
if __name__ == "__main__":
    main()

