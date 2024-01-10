
from openai import OpenAI
import asyncio
client = OpenAI(api_key="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")

def gpt_call(obj):
    obj_string = "Starting position " + obj['position'] + " \nGM Line: "
    for i in range(0,6):
        obj_string += obj['gm_line'][i]['san'] + " " + obj['gm_line'][i]['position'] + " "
    obj_string += "\nStockfish Line: "
    for i in range(0,6):
        obj_string += obj['stockfish_line'][i]['san'] + " " + obj['stockfish_line'][i]['position'] + " "
    #print(obj_string)
    move_analysis = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f'''
                {obj_string}
    Give an output stricly in this format:
    GM Line
    : [Move 1 Category]. [Move 1 Explanation only]
    : [Move 2 Category]. [Move 2 Explanation only]
    : [Move 3 Category]. [Move 3 Explanation only]
    : [Move 4 Category]. [Move 4 Explanation only]
    : [Move 5 Category]. [Move 5 Explanation only]
    : [Move 6 Category]. [Move 6 Explanation only]
    Stockfish Line
    : [Move 1 Category]. [Move 1 Explanation only]
    : [Move 2 Category]. [Move 2 Explanation only]
    : [Move 3 Category]. [Move 3 Explanation only]
    : [Move 4 Category]. [Move 4 Explanation only]
    : [Move 5 Category]. [Move 5 Explanation only]
    : [Move 6 Category]. [Move 6 Explanation only]
    : The move is [Category of Stockfish's first move, but instead of saying the name of the piece, just say "a piece", must be the same as [Move 1 Category] above]
    : Moving a [Piece name of Stockfish's first move] to a [only one of these options:more agressive/more defensive/more strategic] square
    : The [piece] moved ends on {obj['stockfish_line'][0]['uci'][2:4]}
    : [15 words explaining the benefit of the Stockfish move]
    
    First 6 moves are the GM line, then 6 moves of the Stockfish line, then three hints, then the analysis. Analyze each move in the GM and Stockfish lines from the provided chess game, focusing on the tactical and strategic intentions behind each move. Use the position given immediately before each move for your analysis. Classify each move into the following categories and provide a brief explanation of around 15 words for each move. State if a move makes contact with an enemy piece, and state that it puts pressure on the opponent. If it attacks an undefended piece, then state that it attacks the enemy piece, only if that piece is undefended. Order the moves 1, 2, 3, 4, 5, 6. Do not state the name of the move, only the explanation
Include only the classification names before the colon
Giving a Check: Only moves that place the opponent's king in check with a clear tactical or attacking intent (e.g., setting up for a mate or winning material).

Capturing: Moves that remove an opponent's undefended or less valuable piece from the board, not a recapture. Only this classification if there is x in the move san.

Recapturing: Strictly moves that immediately capture a piece or pawn that the opponent has just taken. Only this classification if there is an x in the move san.

Creating a Threat: Moves that directly attack an opponent's higher-value or undefended piece, excluding checks and captures.

Rerouting: (Specific to Knights) Moves that reposition a knight to a more strategically advantageous square, focusing on future potential rather than immediate threats or captures.

Moving out of Danger: Moves specifically made to remove a piece from a square where it can be captured in the next turn, without simultaneously creating a direct counter-threat or capture.

Taking Space: (Pawn Moves Only. Pawn moves that advance beyond the player's half of the board to control more territory, not directly engaging with enemy pawns or pieces.)

Playing a Pawn Break: Pawn moves that directly challenge the opponent’s pawn structure, typically in the center or on the flanks, aiming to alter the pawn dynamics.

Prophylactic Move: Moves primarily made to prevent or hinder a specific opponent's plan or threat, focusing on restricting their piece activity, not directly related to defending against immediate capture threats.

Developing Move: (Non-Knights. Moves that improve the positioning of pieces other than knights, enhancing their effectiveness without immediate threats or captures.)

Improving King Safety: Moves aimed at enhancing the king's safety, such as castling or rearranging pieces to create a defensive structure.

Pawn Structure Improvement: (Non-Advancing Pawns. Moves that adjust or strengthen the pawn structure without advancing into the opponent's half of the board)

Counterattacking: (Direct Response to Threat. Moves that defend against an immediate threat and simultaneously create a direct, immediate counter-threat)
'''}
        ],
        model="gpt-4",
    )
    return move_analysis.choices[0].message.content
if (__name__ == "__main__"):    
    gpt_call({'position': '2r1qrk1/2nnbpp1/bpp1p2p/p2pP3/2PP3P/1P2NNP1/PBQ2PB1/2RR2K1 w - - 5 18', 'stockfish_line': [{'uci': 'e3g4', 'san': 'Ng4', 'position': '2r1qrk1/2nnbpp1/bpp1p2p/p2pP3/2PP2NP/1P3NP1/PBQ2PB1/2RR2K1 b - - 6 18'}, {'uci': 'c8a8', 'san': 'Ra8', 'position': 'r3qrk1/2nnbpp1/bpp1p2p/p2pP3/2PP2NP/1P3NP1/PBQ2PB1/2RR2K1 w - - 7 19'}, {'uci': 'f3e1', 'san': 'Ne1', 'position': 'r3qrk1/2nnbpp1/bpp1p2p/p2pP3/2PP2NP/1P4P1/PBQ2PB1/2RRN1K1 b - - 8 19'}, {'uci': 'b6b5', 'san': 'b5', 'position': 'r3qrk1/2nnbpp1/b1p1p2p/pp1pP3/2PP2NP/1P4P1/PBQ2PB1/2RRN1K1 w - - 0 20'}, {'uci': 'c4d5', 'san': 'cxd5', 'position': 'r3qrk1/2nnbpp1/b1p1p2p/pp1PP3/3P2NP/1P4P1/PBQ2PB1/2RRN1K1 b - - 0 20'}, {'uci': 'c7d5', 'san': 'Nxd5', 'position': 'r3qrk1/3nbpp1/b1p1p2p/pp1nP3/3P2NP/1P4P1/PBQ2PB1/2RRN1K1 w - - 0 21'}], 'gm_line': [{'uci': 'g2f1', 'san': 'Bf1', 'position': '2r1qrk1/2nnbpp1/bpp1p2p/p2pP3/2PP3P/1P2NNP1/PBQ2P2/2RR1BK1 b - - 6 18'}, {'uci': 'c6c5', 'san': 'c5', 'position': '2r1qrk1/2nnbpp1/bp2p2p/p1ppP3/2PP3P/1P2NNP1/PBQ2P2/2RR1BK1 w - - 0 19'}, {'uci': 'f1g2', 'san': 'Bg2', 'position': '2r1qrk1/2nnbpp1/bp2p2p/p1ppP3/2PP3P/1P2NNP1/PBQ2PB1/2RR2K1 b - - 1 19'}, {'uci': 'c5d4', 'san': 'cxd4', 'position': '2r1qrk1/2nnbpp1/bp2p2p/p2pP3/2Pp3P/1P2NNP1/PBQ2PB1/2RR2K1 w - - 0 20'}, {'uci': 'f3d4', 'san': 'Nxd4', 'position': '2r1qrk1/2nnbpp1/bp2p2p/p2pP3/2PN3P/1P2N1P1/PBQ2PB1/2RR2K1 b - - 0 20'}, {'uci': 'b6b5', 'san': 'b5', 'position': '2r1qrk1/2nnbpp1/b3p2p/pp1pP3/2PN3P/1P2N1P1/PBQ2PB1/2RR2K1 w - - 0 21'}]}
)