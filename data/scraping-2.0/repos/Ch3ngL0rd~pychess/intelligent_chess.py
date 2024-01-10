# Script
import re
import json
import openai
import os
from scipy.stats import entropy, norm

import requests
from dotenv import load_dotenv
import chess

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

TEMPERATURE = 0.8
current_fen = "rnbqkbnr/ppp2p1p/8/3p2p1/4Pp1P/5N2/PPPP2P1/RNBQKB1R w KQkq - 0 5"
# uses gpt-3.5-turbo-0613 for intelligent responses

# Two GPT api functions
# 1. is valid move?
# 2. check move strength
# 3. get move suggestions

def getBestMove(fen):
    # Returns the best move, depth + score
    STOCKFISH_PORT = 8002
    SERVER = f"http://127.0.0.1:{STOCKFISH_PORT}"
    ROUTE = "/getAnalysis"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()


def getMaia(fen):
    # Returns the best move and the probability of choosing moves
    MAIA_PORT = 8001
    SERVER = f"http://127.0.0.1:{MAIA_PORT}"
    ROUTE = "/getMove"
    PARAMS = "?fen=" + fen

    request = requests.get(SERVER + ROUTE + PARAMS)
    response = request.json()
    # sorts the moves by probability
    response["maia-1100"]["moves"] = {k: v for k, v in sorted(
        response["maia-1100"]["moves"].items(), key=lambda item: item[1], reverse=True)}
    response["maia-1900"]["moves"] = {k: v for k, v in sorted(
        response["maia-1900"]["moves"].items(), key=lambda item: item[1], reverse=True)}
    return response


def getStockfish(fen, moves):
    # Given a fen and a move, uses stockfish determine whether it is a sound move
    STOCKFISH_PORT = 8002
    SERVER = f"http://127.0.0.1:{STOCKFISH_PORT}"
    ROUTE = "/evaluatePosition"

    PARAMS = {
        "fen": fen,
        "moves": moves
    }

    response = requests.post(SERVER + ROUTE, json=PARAMS)
    return response.json()


def combine_probabilities_and_analysis(maiaEvalMoves, stockfishEval):
    # Combines the move probabilities from Maia and the soundness from Stockfish
    # Returns a dictionary with the move and the combined probability
    combined = {}
    for move in maiaEvalMoves:
        # Merge the two dictionaries
        combined[move] = {
            "probability": maiaEvalMoves[move],
            "win_rate": stockfishEval[move]['win_rate']
        }
    return combined


def get_suggested_moves(fen):
    board = chess.Board()
    board.set_fen(fen)
    maiaEval = getMaia(fen)
    maiaEvalMoves = maiaEval["maia-1100"]["moves"]
    stockfishEval = getStockfish(fen, maiaEval["maia-1100"]["moves"])
    combined = combine_probabilities_and_analysis(
        maiaEvalMoves, stockfishEval['analysis'])
    return combined


def openai_call(messages):
    functions = [
        {
            "name": "evaluate_and_suggest_moves",
            "description": "When a user suggests a move, evaluates the move and suggests alternate moves if it is not valid or sound",
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {
                        "type": "string",
                        "description": "The move to check in SAN format"
                    }
                },
                "required": ["move"]
            }
        },
        {
            "name": "suggest_moves",
            "description": "When a user is stuck, suggests moves for them",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "evaluate_trade_pieces",
            "description": "When a user asks if a trade is good, evaluates the trade",
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {
                        "type": "string",
                        "description": "The move to check in SAN format"
                    }
                },
                "required": ["move"]
            }
        }
    ]

    print("\033[94m" + "RESPONSE" + "\033[0m")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
        temperature=TEMPERATURE,
    )
    print(response)

    response_message = response["choices"][0]["message"]
    if response_message.get("function_call"):
        print("Function call detected")
        available_functions = {
            "evaluate_and_suggest_moves": evaluate_and_suggest_moves,
            "suggest_moves": get_suggested_moves_call,
            "evaluate_trade_pieces": trade_pieces
        }
        print(response_message)
        print(response_message["function_call"])
        print(response_message["function_call"]["name"])
        function_name = response_message["function_call"]["name"]
        function_arguments = json.loads(
            response_message["function_call"]["arguments"])
        function = available_functions[function_name]
        function_response = function(**function_arguments)
        if function_name == "evaluate_and_suggest_moves":
            print("\033[91m" + "Function response" + "\033[0m")
            print(function_response)
            if not function_response["is_valid"]:
                messages.append(
                    {"role": "system", "content": f"The move {function_arguments['move']} is not legal. Here are some suggestions: {function_response['moves']}"})
            elif function_response["is_valid"] and function_response["is_sound"]:
                messages.append(
                    {"role": "system", "content": f"The move {function_arguments['move']} is a sound move."})
            else:
                messages.append(
                    {"role": "system", "content": f"The move {function_arguments['move']} is not a sound move. Here are some suggestions: {function_response['moves']}"})
        elif function_name == "suggest_moves":
            print("\033[91m" + "Function response" + "\033[0m")
            print(function_response)
            messages.append(
                {"role": "system", "content": f"Here are some move suggestions: {function_response['moves']}. Do not explicitly tell the student the moves, but rather give them hints about what to do."})

        elif function_name == "evaluate_trade_pieces":
            print("\033[91m" + "Function response" + "\033[0m")
            print(function_response)
            if not function_response['is_valid']:
                messages.append(
                    {"role": "system", "content": f"The move {function_arguments['move']} is not legal."})
            else:
                messages.append(
                    {"role": "system", "content": f"The move {function_arguments['move']} is {'sound' if function_response['is_sound'] else 'not sound'}. Here is the reasoning which should be explained to the student: "
                     + f"{'Trading a higher value piece' if function_response['higher_value'] else ''}" +
                     f"{'Trading passive piece for active' if function_response['passive_for_active'] else ''}" +
                     f"{'Trading is good when ahead in material' if function_response['ahead_in_material'] else ''}"
                     })
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=messages,
            temperature=TEMPERATURE,
        )
        response_message = response["choices"][0]["message"]
        print(response_message)
        messages.append(response_message)
    else:
        print(response_message)
        messages.append(response_message)

    return messages


def is_valid_move(fen, move):
    print(f"Checking if {move} is valid")
    board = chess.Board()
    board.set_fen(fen)
    try:
        board.push_san(move)
        return True
    except ValueError:
        return False


def evaluate_and_suggest_moves(move):
    global current_fen
    fen = current_fen
    # First checks if the move is valid
    is_valid = is_valid_move(fen, move)

    suggested_moves = get_suggested_moves(fen)
    # Get the top 3 moves sorted by maxmising win_rate * probability
    best_moves = sorted(suggested_moves.items(
    ), key=lambda x: x[1]["win_rate"] * x[1]["probability"], reverse=True)[:3]
    # converts moves from uci to san with probability and win_rate
    moves = []
    for movez in best_moves:
        board = chess.Board()
        board.set_fen(fen)
        san_move = chess.Move.from_uci(movez[0])
        san_move = board.san(san_move)
        san_move = parse_chess_move(san_move)
        moves.append(
            f"{san_move} (chance_chosen: {movez[1]['probability'] * 100:.2f}%, win_rate: {movez[1]['win_rate'] * 100:.2f}%)")

    if not is_valid:
        return {
            "is_valid": False,
            "is_sound": False,
            "moves": moves
        }
    current_win_rate = getWinRate(fen)
    # Checks if the move is sound
    board = chess.Board()
    board.set_fen(fen)
    board.push_san(move)
    new_fen = board.fen()
    new_win_rate = 1 - getWinRate(new_fen)
    is_sound = new_win_rate > current_win_rate - 0.05

    if is_sound:
        return {
            "is_valid": True,
            "is_sound": True,
            "moves": moves
        }

    return {
        "is_valid": True,
        "is_sound": False,
        "moves": moves
    }


def get_suggested_moves_call():
    global current_fen
    fen = current_fen
    suggested_moves = get_suggested_moves(fen)
    best_moves = sorted(suggested_moves.items(
    ), key=lambda x: x[1]["win_rate"] * x[1]["probability"], reverse=True)[:3]
    # converts moves from uci to san with probability and win_rate
    moves = []
    for movez in best_moves:
        board = chess.Board()
        board.set_fen(fen)
        san_move = chess.Move.from_uci(movez[0])
        san_move = board.san(san_move)
        san_move = parse_chess_move(san_move)
        moves.append(
            f"{san_move} (chance_chosen: {movez[1]['probability'] * 100:.2f}%, win_rate: {movez[1]['win_rate'] * 100:.2f}%)")
    return {
        "moves": moves
    }


def trade_pieces(move):
    # First checks if the move is valid
    global current_fen
    fen = current_fen
    is_valid = is_valid_move(fen, move)
    if not is_valid:
        return {
            "is_valid": False,
            "is_sound": False,
            "higher_value": False,
            "passive_for_active": False,
            "ahead_in_material": False
        }

    # Gets which pieces are being traded
    board = chess.Board()
    board.set_fen(fen)
    san_move = board.parse_san(move)
    piece_moved = board.piece_at(san_move.from_square)
    piece_captured = board.piece_at(san_move.to_square)

    piece_values = {"p": 1,  "n": 3,  "b": 3, "r": 5, "q": 9, "k": 0}
    piece_value_moved = piece_values[piece_moved.symbol().lower()]
    piece_value_captured = piece_values[piece_captured.symbol().lower()]

    higher_value = piece_value_moved < piece_value_captured

    # Piece activity using NNUE if equal value
    nnueEval = getNNUE(fen)['nnue_piece_values']
    # gets uci of move
    move_uci = san_move.uci()
    move_square = move_uci[:2]
    caputure_square = move_uci[2:]

    # absolute since black is negative
    passive_for_active = abs(nnueEval[move_square]['value']) < abs(nnueEval[caputure_square]['value'])

    # Removing a piece attacking your king is a good trade
    # TODO: Implement this

    # If ahead in material, simplifying the position is a good trade
    # If win rate is greater than 0.8
    current_win_rate = getWinRate(fen)
    ahead_in_material = current_win_rate > 0.8

    # Checks if the move is sound
    board = chess.Board()
    board.set_fen(fen)
    board.push_san(move)
    new_fen = board.fen()
    new_win_rate = 1 - getWinRate(new_fen)
    is_sound = new_win_rate > current_win_rate - 0.05

    return {
        "is_valid": True,
        "is_sound": is_sound,
        "higher_value": higher_value,
        "passive_for_active": passive_for_active,
        "ahead_in_material": ahead_in_material
    }


def getNNUE(fen):
    # Returns classical heuristic score
    # Returns NNUE score of each piece value
    STOCKFISH_PORT = 8002
    SERVER = f"http://127.0.0.1:{STOCKFISH_PORT}"
    ROUTE = "/getEval"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()


def getWinRate(fen):
    # Calls getBestMove and returns the win rate
    bestMove = getBestMove(fen)
    return bestMove["win_rate"]


def parse_chess_move(move):
    global current_fen
    fen = current_fen
    # Regular expressions for each part
    piece_move = re.compile(
        r'(?P<piece>[KQRBN])(?P<departure>[a-h]?[1-8]?)(?P<capture>x)?(?P<destination>[a-h][1-8])')
    pawn_move = re.compile(
        r'(?P<departure>[a-h]?[1-8]?)(?P<capture>x)?(?P<destination>[a-h][1-8])=?(?P<promotion>[QRBN]?)')
    castle = re.compile(r'(?P<castle>O-O-O|O-O)')
    game_over = re.compile(r'(?P<gameover>1-0|0-1|1/2-1/2)')
    check = re.compile(r'\+')
    checkmate = re.compile(r'#')

    # Dictionaries for translation
    piece_dict = {'K': 'king', 'Q': 'queen',
                  'R': 'rook', 'B': 'bishop', 'N': 'knight',
                  'k': 'king', 'q': 'queen',
                  'r': 'rook', 'b': 'bishop', 'n': 'knight', 'p': 'pawn', 'P': 'pawn'}

    gameover_dict = {'1-0': 'white wins',
                     '0-1': 'black wins', '1/2-1/2': 'draw'}

    # Load the game state
    board = chess.Board()
    board.set_fen(fen)

    # Check each pattern and return the first match
    if piece_match := piece_move.match(move):
        destination = piece_match.group('destination')
        captured_piece = board.piece_at(chess.parse_square(destination))
        return f"{piece_dict[piece_match.group('piece')]} {piece_match.group('capture') and 'captures' or 'moves to'} {captured_piece and piece_dict[str(captured_piece)] or ''} on {destination}"
    elif pawn_match := pawn_move.match(move):
        destination = pawn_match.group('destination')
        captured_piece = board.piece_at(chess.parse_square(destination))
        promotion = pawn_match.group('promotion')
        promotion_phrase = f'promotes to {piece_dict[promotion]}' if promotion else ''
        return f"Pawn {pawn_match.group('capture') and 'captures' or 'moves to'} {captured_piece and piece_dict[str(captured_piece)] or ''} on {destination} {promotion_phrase}"
    elif castle_match := castle.match(move):
        return f"Castle {'queenside' if castle_match.group('castle') == 'O-O-O' else 'kingside'}"
    elif gameover_match := game_over.match(move):
        return gameover_dict[gameover_match.group('gameover')]
    elif check.match(move):
        return "check"
    elif checkmate.match(move):
        return "checkmate"
    else:
        return "Invalid move"


board = chess.Board()
question = "What should I do?"

board.set_fen(current_fen)
best_move = "Bxf4"
predicted_win_rate = getWinRate(current_fen)

maiaEval = getMaia(current_fen)
moves = maiaEval["maia-1100"]["moves"]
stockfishEval = getStockfish(current_fen, maiaEval["maia-1100"]["moves"])
combined = combine_probabilities_and_analysis(moves, stockfishEval['analysis'])
# Converts moves into san format
san_moves = []
# English, san and win rate
for move in combined:
    board = chess.Board()
    board.set_fen(current_fen)
    san_move = chess.Move.from_uci(move)
    san_move = board.san(san_move)
    english_move = parse_chess_move(san_move)
    # move_quality - Good, Bad, Average - difference in predicted win rate and current win rate
    change_in_win_rate = combined[move]["win_rate"] - predicted_win_rate
    if change_in_win_rate > -0.05:
        move_quality = "Good"
    elif change_in_win_rate < -0.05 and change_in_win_rate > -0.1:
        move_quality = "Average"
    else:
        move_quality = "Bad"
    san_moves.append(
        {
            "english": english_move,
            "san": san_move,
            "move_quality": move_quality,
            "probability": combined[move]["probability"],
        })
# Filters move with probability less than 0.05
san_moves = list(filter(lambda x: x["probability"] > 0.05, san_moves))

# Turns moves into a string
san_moves = ','.join(
    list(map(lambda x: f"{x['english']} (QUALITY:{x['move_quality']}, SAN:{x['san']})", san_moves)))

# Remove double spaces
san_moves = re.sub(' +', ' ', san_moves)

prompt = f"""
BOARD:
{board.turn == True and board or board.mirror()}
FEN: {current_fen}
Student Color: {board.turn == True and "White" or "Black"}
Current win rate: {predicted_win_rate}
COMMON MOVES:{san_moves}
{board.is_check() and "The king is in check." or ""}

You are a chess tutor. As the chess tutor, you will wait for your student to respond.
Only give advice about information you are certain about.
When a student asks a question, we'll give a hint about what move to play first, and then give the answer.
Response:"""

# No board
prompt = f"""
Student Color: {board.turn == True and "White" or "Black"}
Current win rate: {predicted_win_rate}
MOVES:{san_moves}
{board.is_check() and "The king is in check." or ""}

You are a chess tutor. As the chess tutor, you will wait for your student to respond.
Keep your responses concise, conversational and natural."""

print(prompt)

# Your student has asked you the following question: "{question}".
messages = [
    {"role": "system", "content": prompt},
]

while True:
    for message in messages:
        print(message["role"] + ": " + message["content"])
    user_input = input("You: ")
    move = "exd5"
    messages.append({"role": "system", "content": f"The user is highlighting the following move {move}"})   
    messages.append({"role": "user", "content": user_input})
    messages = openai_call(messages)