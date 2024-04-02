from copy import deepcopy
from typing import Tuple

from exceptions import AgentException


class AlphaBetaAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token
        self.depth = 6

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        _, next_move = self.alphabeta(connect4, 1, self.depth, -float('inf'), float('inf'))
        return next_move

    def alphabeta(self, connect4, x: int, d: int, alpha: float, beta: float)  -> Tuple[int, int]:
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
                score, _ = self.alphabeta(copy_connect4, 0, d - 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    move = n_column
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        # Min
        else:
            best_score = 1
            score = 1
            for n_column in connect4.possible_drops():
                copy_connect4 = deepcopy(connect4)
                copy_connect4.drop_token(n_column)
                score, _ = self.alphabeta(copy_connect4, 1, d - 1, alpha, beta)
                if score < best_score:
                    best_score = score
                    move = n_column
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
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
