import random
from copy import deepcopy
from typing import Tuple

from exceptions import AgentException


class MinMaxAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token
        self.depth = 4

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        _, next_move = self.minmax(connect4, 1, self.depth)
        return next_move

    def minmax(self, connect4, x: int, d: int) -> Tuple[int, int]:
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return -1, -1
            elif connect4.wins is None:
                return 0, -1
            else:
                return 1, -1
        if d == 0:
            return self.heuristic(connect4), -1
        # Max
        move = -1
        if x == 1:
            best_score = -1
            score = -1
            for n_column in connect4.possible_drops():
                copy_connect4 = deepcopy(connect4)
                copy_connect4.drop_token(n_column)
                score, _ = self.minmax(copy_connect4, 0, d - 1)
                if score > best_score:
                    best_score = score
                    move = n_column
        # Min
        else:
            best_score = 1
            score = 1
            for n_column in connect4.possible_drops():
                copy_connect4 = deepcopy(connect4)
                copy_connect4.drop_token(n_column)
                score, _ = self.minmax(copy_connect4, 1, d - 1)
                if score > best_score:
                    best_score = score
                    move = n_column
        return [best_score, move]

    def heuristic(self, connect4) -> int:
        score = 0
        if connect4.board[0][0] == self.my_token:
            score += 1
        if connect4.board[0][connect4.width - 1] == self.my_token:
            score += 1
        if connect4.board[connect4.height - 1][0] == self.my_token:
            score += 1
        if connect4.board[connect4.height - 1][connect4.width - 1] == self.my_token:
            score += 1
        return score
