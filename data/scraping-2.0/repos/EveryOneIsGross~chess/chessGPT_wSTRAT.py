'''
Currently the strategy mechanic is broken, just an issue converting to embeddings and back, cause chasing arrays is so fun.
'''
import numpy as np
import chess
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import os
import json
import time
import dotenv
import pandas as pd

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Threshold for cosine similarity
SIMILARITY_THRESHOLD = 0.8

# Load the strategems from the JSON files
def load_strategems():
    strategems = {}
    for file_name in os.listdir('strategems'):
        with open(f'strategems/{file_name}', 'r') as f:
            strategem = json.load(f)
            strategems[file_name] = strategem
    return strategems

# Generate embeddings for each state in each strategem
def get_strategem_embeddings(strategems, model):
    strategem_embeddings = {}
    for name, strategem in strategems.items():
        embeddings = [model.encode([state])[0] for state in strategem]
        strategem_embeddings[name] = embeddings
    return strategem_embeddings

def get_best_strategem_move(board, previous_state_embeddings, strategem_embeddings, strategems, model):
    previous_move_embeddings = []  # Initialize it here since you don't use it in your code
    move_scores = rank_moves(board, previous_state_embeddings, previous_move_embeddings, strategem_embeddings, model)
    state_string = str(board)
    state_embedding = model.encode([state_string])[0]

    # Check if the AI won the game and save the embeddings
    if board.is_checkmate() and not board.turn:
        save_ai_game(strategem_embeddings, state_string)

    # Check if embeddings for this strategem already exist
    if board.is_checkmate() and board.turn:
        strategem_name = get_strategem_name(board)
        if strategem_name in strategem_embeddings:
            return get_best_move(move_scores)  # Fallback to the best move based on move scores

    for name, strategem in strategem_embeddings.items():
        similarities = [cosine_similarity([state_embedding], [embedding]) for embedding in strategem]
        if max(similarities) > SIMILARITY_THRESHOLD:
            move_index = similarities.index(max(similarities))
            if move_index + 1 < len(strategems[name]):
                next_state = strategems[name][move_index + 1]
                next_move = get_move_from_state_change(state_string, next_state)
                return next_move

    return get_best_move(move_scores)  # Fallback to the best move based on move scores

def get_strategem_name(board):
    # Create a unique strategem name based on the current board state
    # You can customize this function based on your requirements
    # For example, you can use the current date and time or a random ID
    return f"strategem_{time.time()}"

def save_ai_game(board, strategem_embeddings):
    if board.is_checkmate() and not board.turn:
        # The AI won the game, save the embeddings to a separate strategem file
        strategem_name = get_strategem_name(board)
        if strategem_name not in strategem_embeddings:
            strategem_embeddings[strategem_name] = []
        state_string = str(board)
        state_embedding = model.encode([state_string])[0]
        strategem_embeddings[strategem_name].append(state_embedding)
        with open(f'strategems/{strategem_name}.json', 'w') as f:
            json.dump(strategem_embeddings[strategem_name], f)


def get_move_from_state_change(state1, state2):
    board1 = chess.Board(state1)
    board2 = chess.Board(state2)
    
    # Create a list of differences
    diffs = [i for i in range(64) if board1.piece_at(i) != board2.piece_at(i)]
    
    if len(diffs) != 2:
        raise ValueError('The two states do not appear to be successive states from a legal chess move.')

    # Convert the square numbers to algebraic notation
    move = ''.join(chess.square_name(i) for i in diffs)

    return move

def print_board(board):
    print(str(board))

def rank_moves(board, previous_state_embeddings, previous_move_embeddings, strategem_embeddings, model):
    move_scores = {}
    for move in board.legal_moves:
        state_string = str(board)
        move_string = str(move)
        state_embedding = model.encode([state_string])[0]
        move_embedding = model.encode([move_string])[0]
        state_similarities = []
        move_similarities = []

        if len(previous_state_embeddings) > 0:
            state_similarities = cosine_similarity([state_embedding], previous_state_embeddings)
        if len(previous_move_embeddings) > 0:
            move_similarities = cosine_similarity([move_embedding], previous_move_embeddings)

        average_state_similarity = np.mean(state_similarities) if len(state_similarities) > 0 else 0
        average_move_similarity = np.mean(move_similarities) if len(move_similarities) > 0 else 0
        instruction = f'Here is the current board:\n\n{state_string}\n\nThe average cosine similarity with previous states is {average_state_similarity}. The average cosine similarity with previous moves is {average_move_similarity}.'

        # Initialize instruction_similarity outside the loop
        instruction_similarity = 0

        for name, strategem in strategem_embeddings.items():
            # Update this part to handle 2D embeddings
            similarity = cosine_similarity(
                model.encode([instruction]).reshape(1, -1),
                np.array(strategem).reshape(1, -1)
            )
            instruction_similarity += similarity[0][0]

        move_scores[str(move)] = instruction_similarity + average_state_similarity

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
    sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
    best_move = sorted_moves[0][0]
    return best_move


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
    # Initialize board
    board = chess.Board()

    # Load strategems from JSON files
    strategems = load_strategems()

    # Initialize a dictionary to store the embeddings of the strategems
    strategem_embeddings = {}

    # Convert each strategem into embeddings
    for name, strategem in strategems.items():
        strategem_embeddings[name] = model.encode(strategem)

    # Check the shape of the embeddings
    for name, embeddings in strategem_embeddings.items():
        print(f"The shape of the embeddings for {name}: {np.array(embeddings).shape}")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    board = chess.Board()
    previous_state_embeddings = []
    move_durations = []
    board_positions = []
    moved_pieces = []
    game_result = None

    # Load the strategems and generate their embeddings
    strategems = load_strategems()
    strategem_embeddings = get_strategem_embeddings(strategems, model)

    game_data = load_game_data()

    while True:
        print_board(board)
        if board.is_checkmate():
            print("Checkmate!")
            game_result = "Checkmate"
            # Save the AI-won game for strategems
            save_ai_game(board, strategem_embeddings)
            break
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            print("It's a draw!")
            game_result = "Draw"
            break
        elif board.turn:
            move_start_time = time.time()
            move = get_user_move(board)
            if move is None:
                game_result = "Resign"
                break
            move_durations.append(time.time() - move_start_time)
        else:
            move_start_time = time.time()
            move = get_best_strategem_move(board, previous_state_embeddings, strategem_embeddings, strategems, model)
            move_durations.append(time.time() - move_start_time)

        board.push_uci(move)
        state_string = str(board)
        state_embedding = model.encode([state_string])[0]
        board_positions.append(state_string)
        moved_piece = board.piece_at(chess.parse_square(move[2:4]))
        moved_pieces.append(moved_piece.symbol() if moved_piece else None)

        previous_state_embeddings.append(state_embedding)


    new_game_data = pd.DataFrame({
        "state_embeddings": previous_state_embeddings,
        "move_durations": move_durations,
        "board_positions": board_positions,
        "moved_pieces": moved_pieces,
    })
    new_game_data["game_result"] = game_result  # Assign the game_result value to a new column.

    game_data = pd.concat([game_data, new_game_data], ignore_index=True)
    save_game_data(game_data)

if __name__ == "__main__":
    play_game()
