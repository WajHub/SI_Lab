import random
from copy import deepcopy
from typing import Tuple

from exceptions import AgentException


class MinMaxHeuristicAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token
        self.depth = 5

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        score, next_move = self.minmax(connect4, 0, self.depth)
        return next_move

    def minmax(self, connect4, x: int, d: int) -> Tuple[int, int]:
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return -1, -1
            elif connect4.wins is None:
                return 3, -1
            else:
                return 1, -1
        if d == 0:
            return self.heuristic(connect4), -1
        # Max
        for n_column in connect4.possible_drops():
            move = n_column
        if x == 0:
            best_score = -1
            score = -1
            for n_column in connect4.possible_drops():
                copy_connect4 = deepcopy(connect4)
                copy_connect4.drop_token(n_column)
                score, _ = self.minmax(copy_connect4, 1, d - 1)
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
                score, _ = self.minmax(copy_connect4, 0, d - 1)
                if score < best_score:
                    best_score = score
                    move = n_column
        return [best_score, move]

    def heuristic(self, connect4) -> int:
        total = 0
        score = 0
        for center in connect4.center_column():
            total += 1*50
            if center == self.my_token:
                score = score + 1*50
        weights = [0, 5, 15, 50]
        for fours in connect4.iter_fours():
            score_four = 0
            total += 4*50
            opponent = False
            for four in fours:
                if four == self.my_token:
                    score_four += 1
                elif four != '_':
                    opponent = True
            if opponent is False:
                score = score + weights[score_four]
        return score/total
