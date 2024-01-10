import os
import time
import chess
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def ask_gpt(client, move_list, error_str = "", side = "black"):
    try:
        move_str = ' '.join(move_list)
        completion = client.chat.completions.create(
            model = "gpt-4",
            messages = [
                {"role": "system", "content": "You are a chess master that will make good chess moves"},
                {"role": "user", "content":  error_str + move_str + " based on the game moves given, make one chess move on " + side + " side, only return the chess notation without any additional words"}
            ]
        )
        
        reply = completion.choices[0].message.content

        response = reply

        if ('.' in reply):
            temp = reply.split('.')
            response = temp[-1]
        
        if (' ' in reply):
            temp = reply.split(' ')
            response = temp[-1]

        if (len(response) > 7 or response == None or ' ' in response or ('-' in response and 'O' not in response)):
            print(len(response))
            ask_gpt(client, move_list, side = side)

        else:
            try:
                board.push_san(response)
                print(response)
                move_list.append(response)
                return response
            
            except:
                print(response)
                print(move_list)
                print("ChatGPT made an illegal move")
                error_str = response + " was an illegal move! "
                ask_gpt(client, move_list, error_str, side)
        
    except:
        print("Waiting for ChatGPT...")
        time.sleep(10)
        ask_gpt(client, move_list, side = side)


if __name__ == '__main__':  
    client = OpenAI()

    m = []
    player_move = ['e4', 'Nf3', 'Nc3', 'Bc4', 'g4']

    board = chess.Board()
    
    i = 0
    while board.is_checkmate() == False and board.is_stalemate() == False and board.is_insufficient_material() == False and board.is_seventyfive_moves() == False and board.can_claim_draw() == False:
        #m.append(player_move[i])
        #board.push_san(player_move[i])
        if i % 2 == 0:
            ask_gpt(client, m, side = "white")
        else:
            ask_gpt(client, m, side = "black")
        print(i, m)
        print(board)
        i += 1

