# Script
import json
import openai
import os
from scipy.stats import entropy, norm

import requests
from dotenv import load_dotenv
import chess

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def openai_call(prompt: str, temperature: float = 0.5, past_messages: list = []):
    messages = []
    for message in past_messages:
        messages.append({"role": "user", "content": message})
    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        # model="gpt-4",
        messages=messages,
        temperature=temperature,
        n=1,
        stop=None,
    )
    return response.choices[0].message.content.strip()


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


def getBestMove(fen):
    # Returns the best move, depth + score
    STOCKFISH_PORT = 8002
    SERVER = f"http://127.0.0.1:{STOCKFISH_PORT}"
    ROUTE = "/getAnalysis"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()


def getNNUE(fen):
    # Returns classical heuristic score
    # Returns NNUE score of each piece value
    STOCKFISH_PORT = 8002
    SERVER = f"http://127.0.0.1:{STOCKFISH_PORT}"
    ROUTE = "/getEval"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()


def getBlunder(fen):
    # Returns the blunder score
    BLUNDER_PORT = 8003
    SERVER = f"http://127.0.0.1:{BLUNDER_PORT}"
    ROUTE = "/getBlunder"
    PARAMS = "?fen=" + fen

    response = requests.get(SERVER + ROUTE + PARAMS)
    return response.json()


def getComplexity(maiaEval):
    # Function to calculate entropy
    def calculate_entropy(distribution):
        # Entropy is the uncertainty of a probability distribution
        # Get the probabilities
        probabilities = list(distribution.values())
        # Calculate and return the entropy
        return entropy(probabilities, base=2)

    # Calculates the complexity of a position (move distribution entropy)
    MAIA_1100_STD = 0.8624
    MAIA_1900_STD = 0.8478
    MAIA_1100_MEDIAN = 3.4772
    MAIA_1900_MEDIAN = 2.9330
    # Calculate the entropy for each position and each Maia version
    entropy_value_1100 = calculate_entropy(maiaEval["maia-1100"]["moves"])
    entropy_value_1900 = calculate_entropy(maiaEval["maia-1900"]["moves"])

    # Calculates the complexity of the position
    complexity_1100 = norm.cdf(
        (entropy_value_1100 - MAIA_1100_MEDIAN) / MAIA_1100_STD)
    complexity_1900 = norm.cdf(
        (entropy_value_1900 - MAIA_1900_MEDIAN) / MAIA_1900_STD)

    return complexity_1100, complexity_1900


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
        } | stockfishEval[move]
    return combined

# First question - "I don't know what to do"

# First prototype - Gives the student a move to play
# Second prototype - First gives a hint, and then gives a move to play
# Third prototype - Prompts the student to think first, gives a hint, and then gives a move to play


fen = "r6r/pppkb1pp/3p4/3P4/4R3/6B1/PPP3PP/R5K1 w - - 1 17"
fen = "r6r/pppk2pp/3p1b2/3P4/4R3/6B1/PPP3PP/4R1K1 w - - 3 18"
fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
fen = "rn1qkbnr/ppp2ppp/3p4/8/3PPpb1/5N2/PPP3PP/RNBQKB1R w KQkq - 1 5"
fen = "r2qkbnr/ppp2ppp/2np4/8/3PPBb1/5N2/PPP3PP/RN1QKB1R w KQkq - 1 6"
fen = "r2qk1nr/ppp1bppp/2np4/8/3PPB2/5B2/PPP3PP/RN1QK2R w KQkq - 1 8"

question = "I don't know what to do"

maiaEval = getMaia(fen)
stockfishEval = getStockfish(fen, maiaEval["maia-1100"]["moves"])
best_move = getBestMove(fen)

analysis = combine_probabilities_and_analysis(
    maiaEval["maia-1100"]["moves"], stockfishEval['analysis'])
current_win_rate = best_move['win_rate']

# First prototype
# Gets the move with the most probability that is also sound

# Sorts moves by maximising (probability * win rate)
best_move = max(
    analysis, key=lambda move: analysis[move]['probability'] * analysis[move]['win_rate'])

# Using the fen, converts the best move to algebraic notation
board = chess.Board(fen)
best_move = board.san(chess.Move.from_uci(best_move))
# Prompt to give move
prompt = f"""
EXAMPLE:
Tutor: Alright, no problem. I see a good move here. You can move your rook on a1 to e1. This way, you double up your rooks on the e-file, which can be very powerful.

BOARD:
{board}
Student Color: {board.turn == True and "White" or "Black"}

You are a chess tutor. Your student has asked you the following question: "{question}". The recommended move is {best_move}. 
What do you say to your student?
Tutor:"""


print(board.turn)
# Prompt with hint
prompt = f"""
EXAMPLE CONVERSATION:
INFO: Recommended move is Rook a1 to e1
Student: I don't know what to do.
Tutor: Take a moment to look at your pieces. Can you identify any piece that could be better positioned?
Student: Well, maybe the rook on a1 isn't really contributing much.
Tutor: Good observation. Now, remember how rooks can be more powerful when they're aligned on the same file. Keeping that in mind, where could we move the rook on a1?
Student: Should I move it to e1?
Tutor: Excellent thought! Yes, moving your rook to e1 is a great move. It allows you to double up your rooks on the e-file, which is often a strong setup. Go ahead with that move.

BOARD:
{board}
Student Color: {board.turn == True and "White" or "Black"}

You are a chess tutor. Your student has asked you the following question: "{question}". The recommended move is {best_move}.
As the chess tutor, you will wait for your student to respond.
Let's give a hint about what move to play first, and then give the answer. What do you say to your student?
Tutor:"""

print(prompt)

response = openai_call(prompt)
print(response)

past_messages = [prompt, response]
while True:
    user_input = input("You: ")
    past_messages.append(user_input)
    response = openai_call(user_input, past_messages=past_messages)
    print(response)
    past_messages.append(response)