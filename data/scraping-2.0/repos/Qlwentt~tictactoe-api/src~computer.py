import openai
import os
import json
import copy
from constants.constants import TALLYS, VALID_COORDINATES_TO_POSITIONS, VALID_POSITIONS
from enums.difficulties import Difficulties

openai.api_key = os.environ.get("OPENAI_API_KEY")


class Computer:
    def __init__(self,difficulty, player, opponent):
        self.difficulty = difficulty
        self.memo = {}
        self.choice = None
        self.player = player
        self.opponent = opponent

    def _toTuple(self, board):
        return tuple(map(tuple, board.state))

    def _getOpenAIMove(self,board):
        response = openai.Completion.create(
            model="davinci:ft-personal-2022-07-08-18-57-36",
            prompt=f'{board.state}',
            temperature=0,
            max_tokens=5,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["]\n"]
            )
        
        move = response.choices[0].text.strip()
        move = json.loads(move)
        return VALID_COORDINATES_TO_POSITIONS[tuple(move)]

    def getMove(self, board):
        if self.difficulty == Difficulties.HARD.value:
            return self._getOpenAIMove(board)
        else:
            self._minimax(board, 0, self.player)
            return self.choice

    def _chooseMoveForDifficulty(self, movesAndScores, player):
        if self.difficulty == Difficulties.EASY.value:
            return self._getEasyMove(movesAndScores, player)
        elif self.difficulty == Difficulties.MEDIUM.value:
            return self._getMediumMove(movesAndScores)
        elif self.difficulty == Difficulties.IMPOSSIBLE.value:
            return self._getImpossibleMove(movesAndScores, player)

    def _getEasyMove(self, movesAndScores, player):
        if player == self.player:
            return min(movesAndScores)
        else:
            return max(movesAndScores)

    def _getMediumMove(self, movesAndScores):
        movesAndScores = sorted(movesAndScores, key=lambda x: x[0])
        return movesAndScores[len(movesAndScores)//2]

    def _getImpossibleMove(self, movesAndScores, player):
        if player == self.player:
            return max(movesAndScores)
        else:
            return min(movesAndScores)

    def _score(self, depth,winner):
        if winner:
            if winner == self.player:
                return 10 - depth
            else:
                return -10 + depth
        else:
            return 0
    
    def _minimax(self, board, depth, player):
        winner = board.getWinner()
        if board.isFull() or winner:
            return self._score(depth, winner)
        
        scoresAndMoves = [] # (score, move)
        depth += 1

        for move in board.getValidMoves():
            theoreticalBoard = copy.deepcopy(board)
            theoreticalBoard.markMove(VALID_POSITIONS[move], player, TALLYS[player])
            if player == self.player:
                newPlayer = self.opponent
            else:
                newPlayer = self.player
            if (self._toTuple(theoreticalBoard), newPlayer) in self.memo:
                score = self.memo[(self._toTuple(theoreticalBoard), newPlayer)]
            else:
                score = self._minimax(theoreticalBoard, depth, newPlayer)
                self.memo[(self._toTuple(theoreticalBoard), newPlayer)] = score
            scoresAndMoves.append((score, move)) 
                        
        score, move = self._chooseMoveForDifficulty(scoresAndMoves, player)
        self.choice = move
        return score