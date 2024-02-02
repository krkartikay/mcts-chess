import chess
import random
import torch
import numpy as np

import mcts
from mcts import expand_node, mcts_choose_move, MCTSNode
from convert import board_to_tensor, moves_to_tensor
from action import action_to_move

from typing import List, Tuple

from model import ChessModel, get_model

mcts.N_SIM = 200
mcts.SAMPLING_TEMPERATURE = 5


def generate_games(num_games: int):
    model = get_model()
    games = []
    print(f"Generating games.")
    for i in range(num_games):
        print(f"=================== GAME {i} ===================.")
        game = generate_selfplay_game(model)
        games.append(game)
    print(f"Done! Generated {num_games} games!")

    print("Converting data to tensors.")
    positions, valid_moves, values = convert_to_tensors(games)

    print(f"Saving to output file. Shape:")
    print(f"positions : {positions.size()}")
    print(f"moves     : {valid_moves.size()}")
    print(f"moves     : {values.size()}")
    print()

    save_to_file(positions, valid_moves, values)


def generate_selfplay_game(model: ChessModel) -> List[Tuple[chess.Board, torch.Tensor, torch.Tensor]]:
    board = chess.Board()
    history: List[Tuple[chess.Board, torch.Tensor, torch.Tensor]] = []
    root_node = MCTSNode()
    # Need to do this the first time
    _value = expand_node(root_node, board, model)
    while not (board.is_game_over() or board.is_fifty_moves()):
        selected_action, move_probs, current_eval = mcts_choose_move(
            root_node, board, model)
        selected_move = action_to_move(selected_action, board)
        print(board, "\n", selected_move, current_eval)
        # print(selected_move, move_probs, current_eval)
        # top_actions = top_actions = torch.argsort(
        #     move_probs)[-20:].cpu().numpy()[::-1]
        # for action in top_actions:
        #     print(action_to_move(int(action), board),
        #           move_probs[action])

        history.append((chess.Board(board.fen()), move_probs, current_eval))
        board.push(selected_move)
        if selected_action not in root_node.next_states:
            print("Terminal value: ",
                  root_node.terminal_states[selected_action])
        else:
            root_node = root_node.next_states[selected_action]
    print(board.outcome())
    return history


def convert_to_tensors(
        games: List[List[Tuple[chess.Board, torch.Tensor, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_positions = []
    all_valid_moves = []
    all_values = []
    for game in games:
        for position, move_probs, value in game:
            board_tensor = board_to_tensor(position)
            moves_tensor = move_probs
            all_positions.append(board_tensor)
            all_valid_moves.append(moves_tensor)
            all_values.append(value)
    positions = torch.stack(all_positions)
    valid_moves = torch.stack(all_valid_moves)
    values = torch.stack(all_values)
    return positions, valid_moves, values


def save_to_file(positions, moves, values, filename='games.pth'):
    with open(filename, 'wb') as datafile:
        torch.save({"positions": positions, "moves": moves,
                   "values": values}, datafile)


if __name__ == "__main__":
    generate_games(10)
