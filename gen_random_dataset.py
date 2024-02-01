import chess
import random
import torch

from convert import board_to_tensor, moves_to_tensor

from typing import List, Tuple


def generate_games(num_games: int):
    games = []
    print(f"Generating games.")
    for i in range(num_games):
        print(f"Game {i}")
        game = generate_random_game()
        games.append(game)
    print(f"Done! Generated {num_games} games!")

    print("Converting data to tensors.")
    positions, valid_moves, values = convert_to_tensors(games)

    print(f"Saving to output file. Shape:")
    print(f"positions : {positions.size()}")
    print(f"moves     : {valid_moves.size()}")
    print(f"values     : {values.size()}")
    print()

    save_to_file(positions, valid_moves, values)


def generate_random_game() -> List[Tuple[chess.Board, List[chess.Move], float]]:
    board = chess.Board()
    history: List[Tuple[chess.Board, List[chess.Move]]] = []
    while not board.is_game_over():
        # print(board)
        valid_moves = list(board.generate_legal_moves())
        # print(valid_moves)
        random_move = random.choice(valid_moves)
        history.append((chess.Board(board.fen()), valid_moves))
        board.push(random_move)
    outcome = board.outcome()
    assert outcome is not None
    history_with_values: List[Tuple[chess.Board, List[chess.Move], float]] = []
    for board, moves in history:
        if outcome.termination == chess.Termination.CHECKMATE:
            value = 1 if board.turn == outcome.winner else -1
        else:
            value = 0
        history_with_values.append((board, moves, value))
    return history_with_values


def convert_to_tensors(
        games: List[List[Tuple[chess.Board, List[chess.Move], float]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_positions = []
    all_valid_moves = []
    all_values = []
    for game in games:
        for position, valid_moves, value in game:
            board_tensor = board_to_tensor(position)
            moves_tensor = moves_to_tensor(valid_moves)
            all_positions.append(board_tensor)
            all_valid_moves.append(moves_tensor)
            all_values.append(value)
    print(all_values)
    positions = torch.stack(all_positions)
    valid_moves = torch.stack(all_valid_moves)
    values = torch.tensor(all_values).float()
    print(values)
    return positions, valid_moves, values


def save_to_file(positions, moves, values, filename='games.pth'):
    with open(filename, 'wb') as datafile:
        torch.save({"positions": positions, "moves": moves,
                   "values": values}, datafile)


if __name__ == "__main__":
    generate_games(100)
