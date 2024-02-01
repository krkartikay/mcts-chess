import torch
import torch.nn as nn
import torch.nn.functional as F

N_HIDDEN = 4096


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(7*8*8, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, 64*64),
        )

    def forward(self, x):
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Apply layers
        x = self.layers(x)
        # Outputs the logits normalised by log_softmax
        x = F.log_softmax(x, dim=1)
        return x

    def device(self):
        return next(self.parameters()).device


def get_model() -> ChessModel:
    # Create the neural net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = torch.load(open("saved_model.pth", "rb"))
    except FileNotFoundError:
        model = ChessModel()
    return model.to(device)


if __name__ == "__main__":
    model = ChessModel()
    print(model)
