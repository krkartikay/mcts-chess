import torch
import chess
import chess.svg
import numpy as np
from action import action_to_move
from matplotlib import pyplot as plt

ARROW_WEIGHT = 1000


def plot_pos_tensor(tensor: torch.Tensor):
    fig, axs = plt.subplots(1, 7, figsize=(15, 5))
    channel_names = ['TURN', 'PAWN', 'KNIGHT',
                     'BISHOP', 'ROOK', 'QUEEN', 'KING']
    for i in range(7):
        axs[i].imshow(tensor[i, :, :], cmap='gray',
                      vmin=-1, vmax=1, origin="lower")
        axs[i].set_title(channel_names[i])
    plt.show()


def plot_move_set(moves: torch.Tensor, filename="move_set.png"):
    moves_view = np.zeros((64, 64))
    for start_row in range(8):
        for start_col in range(8):
            for end_row in range(8):
                for end_col in range(8):
                    action_num = (start_row+start_col*8)*64+(end_row+end_col*8)
                    moves_view[start_col*8+end_col][start_row *
                                                    8+end_row] = moves[action_num]
    plt.imshow(moves_view, origin="lower")
    plt.savefig(filename)
    plt.close()


def plot_board_moves(board: chess.Board, probs: torch.Tensor, k=100, size=400):
    arrows = []
    probs, actions = torch.topk(probs, k)
    for prob, action in zip(probs, actions):
        arrow_weight = min(int(prob*ARROW_WEIGHT), 256)
        arrow_color = f"#009900{arrow_weight:02x}"
        move = action_to_move(action.item(), board)
        arrows.append(chess.svg.Arrow(move.from_square,
                      move.to_square, color=arrow_color))
    return chess.svg.board(board, arrows=arrows, size=size)
