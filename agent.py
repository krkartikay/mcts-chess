import chess
import torch
import random

from mcts import MCTSNode, mcts_choose_move, expand_node
from action import action_to_move, move_to_action
from model import get_model, neural_net_eval

TEMPERATURE = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChessAgent:
    def choose_move(self) -> chess.Move:
        raise NotImplementedError

    def update_position(self, move: chess.Move) -> None:
        raise NotImplementedError


class RandomChessAgent(ChessAgent):
    def __init__(self):
        self.board = chess.Board()

    def choose_move(self) -> chess.Move:
        # Doing this hackery to remove underpromotions
        return random.choice([action_to_move(move_to_action(m), self.board) for m in self.board.legal_moves])

    def update_position(self, move: chess.Move) -> None:
        self.board.push(move)

    def reset(self):
        self.board = chess.Board()


class MCTSAgent(ChessAgent):
    def __init__(self, model):
        self.board = chess.Board()
        self.root_node = MCTSNode()
        self.model = model
        expand_node(self.root_node, self.board, self.model)

    def choose_move(self) -> chess.Move:
        chosen_action, _, _ = mcts_choose_move(
            self.root_node, self.board, self.model)
        return action_to_move(chosen_action, self.board)

    def update_position(self, move: chess.Move) -> None:
        self.board.push(move)
        action = move_to_action(move)
        if action in self.root_node.next_states:
            self.root_node = self.root_node.next_states[action]
        else:
            # print("Next node did not exist!!")
            self.root_node = MCTSNode()
            expand_node(self.root_node, self.board, self.model)


class NNetAgent(ChessAgent):
    def __init__(self):
        self.board = chess.Board()
        self.model = get_model()

    def choose_move(self) -> chess.Move:
        probs, value = neural_net_eval(self.board, self.model)
        probs = probs.to(device)
        legal_mask = torch.zeros(64*64).to(device)
        for move in self.board.legal_moves:
            legal_mask[move_to_action(move)] = 1
        probs *= legal_mask
        # print(probs.sum())
        if (probs.sum() <= 1e-8):
            # What to do then?
            # print(probs)
            # print(self.board)
            # print(self.board.fen())
            probs = legal_mask
        action = int(torch.multinomial(probs ** TEMPERATURE, 1).item())
        return action_to_move(action, self.board)

    def update_position(self, move: chess.Move) -> None:
        self.board.push(move)
        action = move_to_action(move)

    def reset(self):
        self.board = chess.Board()
