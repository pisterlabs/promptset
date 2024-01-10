import chess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import json
import time
import dotenv
import os
import pandas as pd

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def print_board(board):
    print(str(board))

def rank_moves(board, previous_state_embeddings, previous_move_embeddings, model):
    moves = list(board.legal_moves)
    move_scores = []
    for move in moves:
        board.push(move)
        state_string = str(board)
        move_string = move.uci()
        state_embedding = model.encode([state_string])[0]
        move_embedding = model.encode([move_string])[0]
        state_similarities = cosine_similarity([state_embedding], previous_state_embeddings)[0]
        move_similarities = cosine_similarity([move_embedding], previous_move_embeddings)[0]
        average_state_similarity = np.mean(state_similarities)
        average_move_similarity = np.mean(move_similarities)
        instruction = f'Here is the current board:\n\n{state_string}\n\nThe average cosine similarity of the current state to the previous states is {average_state_similarity} and the average cosine similarity of the current move to the previous moves is {average_move_similarity}.\n\nPlease provide a score for this position. ...'
        score = get_openai_score(instruction)
        move_scores.append((move, score))
        board.pop()
    return move_scores

def get_openai_score(instruction):
    chess_rules = '\n1. The game is played on an 8x8 grid, with alternating white and black squares. \
        \n2. Each player starts with 16 pieces: one king, one queen, two rooks, two knights, two bishops, and eight pawns. \
        \n3. The goal of the game is to checkmate the opponent\'s king. This means the opponent\'s king is in a position to be captured ("in check") and there is no way to move the king out of capture ("checkmate"). \
        \n4. The game can also end by resignation. If a player decides they cannot win, they can choose to resign, ending the game immediately. \
        \n5. The game is drawn if neither player can checkmate the other\'s king. This can occur under several conditions, including insufficient material to checkmate, stalemate, or threefold repetition of a position.'
    prompt = chess_rules + "\n\n" + instruction
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=10)
    score = response.choices[0].text.strip()
    return score

def get_best_move(move_scores):
    move_scores.sort(key=lambda x: x[1], reverse=True)
    best_move = move_scores[0][0]
    return best_move.uci()

def get_openai_move(board, previous_state_embeddings, previous_move_embeddings, model):
    move_scores = rank_moves(board, previous_state_embeddings, previous_move_embeddings, model)
    best_move = get_best_move(move_scores)
    return best_move

def get_user_move(board):
    while True:
        move = input("Enter your move: ")
        if move.lower() == 'resign':
            print("You've resigned. Game over.")
            return None
        try:
            move_obj = chess.Move.from_uci(move)
            if move_obj in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except:
            print("Invalid move. Try again.")

def load_game_data():
    try:
        with open('game_data.json', 'r') as f:
            return pd.read_json(f, lines=True)  # Specify `lines=True` to read JSON records as lines
    except FileNotFoundError:
        return pd.DataFrame(columns=['state_embeddings', 'move_embeddings', 'move_durations', 'board_positions', 'moved_pieces'])

def save_game_data(game_data):
    game_data.to_json('game_data.json', orient='records', lines=True)

def play_game():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    board = chess.Board()
    previous_state_embeddings = []
    previous_move_embeddings = []
    move_durations = []
    board_positions = []
    moved_pieces = []
    game_result = None  # variable to hold game result

    # Load existing game data
    game_data = load_game_data()

    while True:
        print_board(board)
        if board.is_checkmate():
            print("Checkmate!")
            game_result = "Checkmate"
            break
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            print("It's a draw!")
            game_result = "Draw"
            break
        elif board.turn:
            move_start_time = time.time()  # Record move start time
            move = get_user_move(board)
            if move is None:  # user resigns
                game_result = "Resign"
                break
            move_durations.append(time.time() - move_start_time)  # Record move duration
        else:
            move_start_time = time.time()  # Record move start time
            move = get_openai_move(board, previous_state_embeddings, previous_move_embeddings, model)
            move_durations.append(time.time() - move_start_time)  # Record move duration

        board.push_uci(move)
        state_string = str(board)
        move_string = move
        state_embedding = model.encode([state_string])[0]
        move_embedding = model.encode([move_string])[0]
        board_positions.append(state_string)  # Record board position

        # Record if any piece has moved from its starting position
        moved_piece = board.piece_at(chess.parse_square(move[2:4]))
        moved_pieces.append(moved_piece.symbol() if moved_piece else None)

        previous_state_embeddings.append(state_embedding)
        previous_move_embeddings.append(move_embedding)

    # Save the new game data and append it to the existing DataFrame
    new_game_data = pd.DataFrame({
        "state_embeddings": previous_state_embeddings,
        "move_embeddings": previous_move_embeddings,
        "move_durations": move_durations,
        "board_positions": board_positions,
        "moved_pieces": moved_pieces,
        "game_result": game_result  # add game_result
    })

    if not new_game_data.empty:  # Check if the new_game_data DataFrame is not empty
        game_data = pd.concat([game_data, new_game_data], ignore_index=True)

    # Save the updated game data
    save_game_data(game_data)

if __name__ == "__main__":
    play_game()
