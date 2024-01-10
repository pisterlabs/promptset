# Script
import json
from matplotlib import pyplot as plt
import openai
import os
from scipy.stats import entropy, norm

import requests
from dotenv import load_dotenv
import chess

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

# Function to calculate entropy
def calculate_entropy(distribution):
    # Entropy is the uncertainty of a probability distribution
    # Get the probabilities
    probabilities = list(distribution.values())
    # Calculate and return the entropy
    return entropy(probabilities, base=2)


def getComplexity(maiaEval):
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
            "win_rate": stockfishEval[move]["win_rate"]
        }
    return combined

fen = "rn1qkbnr/ppp2ppp/3p4/8/3PPpb1/5N2/PPP3PP/RNBQKB1R w KQkq - 1 5"
maiaEval = getMaia(fen)
nnueEval = getNNUE(fen)
blunderEval = getBlunder(fen)
complexity_1100, complexity_1900 = getComplexity(maiaEval)
stockfishEval = getStockfish(fen, maiaEval["maia-1100"]["moves"])
 
combined = combine_probabilities_and_analysis(maiaEval["maia-1100"]["moves"], stockfishEval["analysis"])

moves = maiaEval["maia-1100"]["moves"]

best_move = getBestMove(fen)

print(maiaEval)
print(stockfishEval)
print(nnueEval)
print(blunderEval)
print(best_move)
print(f"Complexity: {complexity_1100}")
print(f"Complexity: {complexity_1900}")
print(combined)