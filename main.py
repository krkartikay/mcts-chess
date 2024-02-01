import torch
import train
import torch.nn.functional as F
from model import get_model
from train import load_data, train_model
from convert import tensor_to_board
from visualization import plot_move_set, plot_board_moves

device = 'cuda' if torch.cuda.is_available() else 'cpu'
positions, valid_moves = load_data()
positions = positions.to(device)
valid_moves = valid_moves.to(device)

train.NUM_EPOCHS = 1

chess_model = get_model().to(device)

IDX = 181


def main():
    checkpoint(chess_model)

    for i in range(10):
        result = train_model(chess_model, positions, valid_moves)
        checkpoint(chess_model)
        print(result)


def checkpoint(model):
    torch.save(model, open("saved_model.pth", "wb"))
    with torch.no_grad():
        move_logits = model(positions[[IDX]])[0]
        plot_move_set(move_logits, filename="move_logits.png")
        move_probs = F.softmax(move_logits, dim=0)
        plot_move_set(move_probs, filename="move_probs.png")
        board_moves = plot_board_moves(
            tensor_to_board(positions[IDX]), move_probs)
        with open("board_moves.svg", "w") as svg_file:
            svg_file.write(board_moves)
            svg_file.flush()


if __name__ == "__main__":
    main()
