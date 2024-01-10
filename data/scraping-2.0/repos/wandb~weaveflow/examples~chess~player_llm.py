import weave
import chess
import random
import re

from player import Player


@weave.type()
class LLMPlayer(Player):
    max_steps: int
    system_message: str
    model_name: str = "gpt-4"

    @weave.op()
    async def move(self, board_fen: str) -> str:
        import random
        import re
        import chess
        import openai
        from weave.monitoring.openai import patch

        patch()

        client = openai.AsyncOpenAI()

        board = chess.Board(board_fen)

        model_name = self.model_name

        color = "white" if board.turn else "black"
        # legal_moves = [m.uci() for m in board.legal_moves]
        # legal_moves_s = ",".join(legal_moves)
        # prompt = f"The current board state is:\n\n{str(board)}\n\nYou are playing: {color}\n\nYou must choose the next move, in UCI format.\n\nHere are the current legal moves: {legal_moves_s}\n"
        prompt = f"The current board state is:\n\n{str(board)}\n\nYou are playing: {color}\n\nYou must choose the next move, in UCI format.\n\n"
        prompt += "The pieces are laid out as follows\n"
        for piece_type in chess.PIECE_TYPES:
            for piece_color in [chess.WHITE, chess.BLACK]:
                # Get squares where this type of piece for this color is located
                squares = board.pieces(piece_type, piece_color)

                # Iterate over these squares
                for square in squares:
                    piece = board.piece_at(square)
                    prompt += f"Piece {piece.symbol()} at {chess.square_name(square)}\n"
        prompt += "\nPlease output the following: a tree of moves that you have considered with score and analysis of each node, your move in UCI format, and the centi-pawn gain or loss you believe this move has."
        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user",
                "content": prompt,
            },
        ]
        print("LLM PROMPT:", prompt)
        for i in range(self.max_steps):
            response = await client.chat.completions.create(
                model=model_name, messages=messages
            )
            response_message = response.choices[0].message
            if response_message.content is None:
                raise ValueError("Response message content is None")

            # Use regex to extract all UCI moves from the text
            moves: list[str] = re.findall(
                r"\b[a-h][1-8][a-h][1-8][a-z]?\b", response_message.content
            )
            next_move = None
            if moves:
                next_move = moves[-1]

            if next_move is None:
                feedback = "Your response did not contain a valid move in UCI format."
            else:
                try:
                    move = chess.Move.from_uci(next_move)
                    if move in board.legal_moves:
                        print(f"\nCOMPLETE WITH FINAL MOVE: {move}\n\n")
                        return move.uci()
                    else:
                        feedback = f"Your move {move.uci()} is not valid currently."
                except:
                    feedback = f"Couldn't parse your move {next_move} in UCI format."

            messages.append(response_message)
            feedback_message = f"Error: {feedback}\n\nHere is the current board state:\n\n{str(board)}\n"
            print(feedback_message)
            messages.append({"role": "user", "content": feedback_message})

        # Choose a random move from legal_moves
        return random.choice(list(board.legal_moves)).uci()
