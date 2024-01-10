import pandas as pd
from openai import OpenAI
import random
from sklearn.model_selection import train_test_split

def check_win(board):
    """Check if there's a winner on the board."""
    for i in range(3):
        # Check rows and columns
        if board[i][0] == board[i][1] == board[i][2] != None or \
           board[0][i] == board[1][i] == board[2][i] != None:
            return True
    # Check diagonals
    return (board[0][0] == board[1][1] == board[2][2] != None) or \
       (board[0][2] == board[1][1] == board[2][0] != None)

def is_draw(board):
    return all(all(cell is not None for cell in row) for row in board)

def minimax(board, depth, is_maximizing, alpha, beta):
    if check_win(board):
        return depth - 10 if is_maximizing else 10 - depth
    if is_draw(board):
        return 0

    if is_maximizing:  # 'X's turn (Maximizing player)
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = None
                    best_score = max(best_score, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
        return best_score
    else:  # 'O's turn (Minimizing player)
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = None
                    best_score = min(best_score, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
        return best_score

def best_move(board, player):
    """Find all the best moves for 'X'"""
    best_score = -float('inf')
    winning_moves = []
    moves = []

    for i in range(3):
        for j in range(3):
            if board[i][j] is None:
                board[i][j] = 'X'
                score = minimax(board, 0, False, -float('inf'), float('inf'))
                board[i][j] = None
                
                if score == 1:
                    winning_moves.append((i, j, 'X'))
                elif (score > best_score):
                    best_score = score
                    moves = [(i, j, 'X')]
                elif score == best_score:
                    moves.append((i, j, 'X'))

    return winning_moves if len(winning_moves) > 0 else moves

def build_optimal_moves_going_first():
    optimal_moves = {}
    def build_sequences(board, sequence):
        if check_win(board) or is_draw(board):
            return
        
        if len(sequence) % 2 == 0: # X's move - try only the best move
            next_player = 'X'
            next_moves = best_move(board, 'X')
            optimal_moves[tuple(sequence)] = next_moves
            for move in next_moves:
                i, j, player = move
                board[i][j] = player
                new_sequence = sequence.copy()
                new_sequence.append(move)
                build_sequences(board, new_sequence)
                board[i][j] = None

        else: # O's move - try all possible moves
            next_player = 'O'
            for i in range(3):
                for j in range(3):
                    if board[i][j] is None:
                        board[i][j] = next_player
                        new_sequence = sequence.copy()
                        new_sequence.append((i, j, next_player))
                        build_sequences(board, new_sequence)
                        board[i][j] = None        
        
    build_sequences([[None for _ in range(3)] for _ in range(3)], [])
    return optimal_moves

def build_optimal_moves_going_second():
    optimal_moves = {}
    def build_sequences(board, sequence):
        if check_win(board) or is_draw(board):
            return
        
        if len(sequence) % 2 != 0: # O's move - try only the best move
            next_player = 'O'
            next_moves = best_move(board, 'O')
            optimal_moves[tuple(sequence)] = next_moves
            for move in next_moves:
                i, j, player = move
                board[i][j] = player
                new_sequence = sequence.copy()
                new_sequence.append(move)
                build_sequences(board, new_sequence)
                board[i][j] = None

        else: # X's move - try all possible moves
            next_player = 'X'
            for i in range(3):
                for j in range(3):
                    if board[i][j] is None:
                        board[i][j] = next_player
                        new_sequence = sequence.copy()
                        new_sequence.append((i, j, next_player))
                        build_sequences(board, new_sequence)
                        board[i][j] = None        
        
    build_sequences([[None for _ in range(3)] for _ in range(3)], [])
    return optimal_moves

def construct_prompt(k):
    # @type k: tuple of tuples representing previous moves
    # @type v: tuple

    prompt = ''

    for move in k:
        prompt += f'({move[0]}, {move[1]}, {move[2]})\n'

    return prompt[:-1]

def evaluate_output(response, v):
    return response in v

def test_model(items, model_name, file):
    
    correct = 0
    sys_prompt = ''
    with open(file, 'r') as f:
        sys_prompt = f.read()

    temperature = 0.0
    max_tokens = 128

    for k, v in items:
        prompt = construct_prompt(k)    

        client = OpenAI(api_key="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")
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': prompt}
            ],
        )

        response_text = response.choices[0].message.content

        ans = (int(response_text[1]), int(response_text[4]), 'X')
        evaluation = evaluate_output(ans, v)
        correct += 1 if evaluation else 0

    print(f'Correct: {correct}/{len(items)}')
    

def evaluate_gpts():
    optimal_moves_going_first = build_optimal_moves_going_first()
    optimal_moves_going_second = build_optimal_moves_going_second()

    dict_items_list = list(optimal_moves_going_first.items())
    dict_items_list_second = list(optimal_moves_going_second.items())
    tests_first = random.sample(dict_items_list, 500)
    tests_second = random.sample(dict_items_list_second, 500)

    for model_name in ['gpt-4-1106-preview', 'ft:gpt-3.5-turbo-0613:personal::8SjmN8I3']:
        test_model(tests_first, model_name, 'sys_prompt_zero.txt')
        test_model(tests_second, model_name, 'sys_prompt_zero.txt')


if __name__ == '__main__':
    evaluate_gpts()
