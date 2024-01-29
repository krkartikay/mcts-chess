import chess
import random

from mcts import MCTSNode, mcts_choose_move, expand_node
from action import move_to_action


class ChessAgent:
    def choose_move(self) -> chess.Move:
        raise NotImplementedError

    def update_position(self, move: chess.Move) -> None:
        raise NotImplementedError


class RandomChessAgent(ChessAgent):
    def __init__(self):
        self.board = chess.Board()

    def choose_move(self) -> chess.Move:
        return random.choice(list(self.board.legal_moves))

    def update_position(self, move: chess.Move) -> None:
        self.board.push(move)


class MCTSAgent(ChessAgent):
    def __init__(self):
        self.board = chess.Board()
        self.root_node = MCTSNode()
        expand_node(self.root_node, self.board)

    def choose_move(self) -> chess.Move:
        chosen_move, _, _ = mcts_choose_move(self.root_node, self.board)
        return chosen_move

    def update_position(self, move: chess.Move) -> None:
        self.board.push(move)
        action = move_to_action(move)
        if action in self.root_node.next_states:
            self.root_node = self.root_node.next_states[action]
        else:
            self.root_node = MCTSNode()
            expand_node(self.root_node, self.board)
