import chess
import numpy

from typing import List


def move_to_action(move: chess.Move) -> int:
    a = move.from_square
    b = move.to_square
    idx = (a * 64) + b
    return idx


def action_to_move(action: int, board: chess.Board | None) -> chess.Move:
    a, b = divmod(action, 64)
    move = chess.Move(a, b)

    # check for possible promotion
    if (board is not None
            and chess.square_rank(b) == (7 if board.turn == chess.WHITE else 0)
            and board.piece_type_at(a) == chess.PAWN):
        move = chess.Move(a, b, chess.QUEEN)

    return move


def moves_to_tensor(moves: List[chess.Move]) -> numpy.ndarray:
    moves_tensor = numpy.zeros(64*64)
    valid_actions = [move_to_action(move) for move in moves]
    moves_tensor[valid_actions] = 1
    return moves_tensor
