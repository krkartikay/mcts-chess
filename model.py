import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from convert import board_to_tensor

N_HIDDEN = 4096

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(7*8*8, N_HIDDEN),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(N_HIDDEN, 1)
        self.prob_head = nn.Linear(N_HIDDEN, 64*64)

    def forward(self, x):
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Apply layers
        x = self.layers(x)
        logits = self.prob_head(x)
        normalized_logits = F.log_softmax(logits, dim=1)
        value = self.value_head(x)
        normalized_value = F.tanh(value)
        # Outputs the logits normalised by log_softmax
        return (normalized_logits, normalized_value)

    def device(self):
        return next(self.parameters()).device


def get_model() -> ChessModel:
    # Create the neural net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = torch.load(open("saved_model.pth", "rb"))
        print("Loaded model!")
    except FileNotFoundError:
        model = ChessModel()
        print("Creating new model!")
    return model.to(device)


def neural_net_eval(board: chess.Board, model: ChessModel) -> Tuple[torch.Tensor, float]:
    """
    This function is an interface to a neural net that returns two things,
    a policy and an evaluation (between +1 and -1) for the given position.
    """
    with torch.no_grad():
        logits, values = model(board_to_tensor(
            board).unsqueeze(dim=0).to(device))
        return F.softmax(logits[0], dim=0), values.item()


if __name__ == "__main__":
    model = ChessModel()
    print(model)
