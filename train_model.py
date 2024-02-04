import chess
import torch
import training
import torch.nn.functional as F
from model import get_model
from training import load_data, train_model
from convert import board_to_tensor, tensor_to_board
from visualization import plot_move_set, plot_board_moves

IDX = 0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chess_model = get_model().to(device)
    checkpoint(chess_model)
    try:
        positions, valid_moves, values = load_data()
        positions = positions.to(device)
        valid_moves = valid_moves.to(device)
        values = values.to(device)
        run_training(chess_model, positions, valid_moves, values)
    except FileNotFoundError:
        pass


def run_training(chess_model, positions, moves, values):
    result = train_model(chess_model, positions, moves, values)
    checkpoint(chess_model)
    print(result)


def checkpoint(model):
    torch.save(model, open("saved_model.pth", "wb"))
    with torch.no_grad():
        # position = positions[[IDX]]
        position = board_to_tensor(chess.Board()).unsqueeze(dim=0).to('cuda')
        move_logits, value = model(position)
        move_logits = move_logits[0]
        value = value[0]
        print(f"Move eval: {value}")
        plot_move_set(move_logits, filename="move_logits.png")
        move_probs = F.softmax(move_logits, dim=0)
        plot_move_set(move_probs, filename="move_probs.png")
        move_probs /= move_probs.sum()
        board_moves = plot_board_moves(
            tensor_to_board(position[0]), move_probs)
        with open("board_moves.svg", "w") as svg_file:
            svg_file.write(board_moves)
            svg_file.flush()


if __name__ == "__main__":
    main()
