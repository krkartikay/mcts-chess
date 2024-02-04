import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from typing import Dict, List, Tuple

from model import ChessModel
from observer import Observer

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20

loss_observer = Observer('loss', labels=['train_loss', 'test_loss'])


def load_data(filename="games.pth") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Loading data...")

    with open("games.pth", "rb") as datafile:
        data = torch.load(datafile)
        positions: torch.Tensor = data["positions"]
        valid_moves: torch.Tensor = data["moves"]
        values: torch.Tensor = data["values"]

    print("Loaded data. Shape: ")
    print(f"positions : {positions.size()}")
    print(f"moves     : {valid_moves.size()}")
    print(f"values     : {values.size()}")

    return positions, valid_moves, values


def train_model(model: ChessModel, positions: torch.Tensor, valid_moves: torch.Tensor, values: torch.Tensor) -> Dict:
    # Splitting the dataset into training and testing
    train_size = int(0.8 * len(positions))  # 80% for training
    test_size = len(positions) - train_size

    train_dataset, test_dataset = random_split(
        TensorDataset(positions, valid_moves, values), [train_size, test_size])

    dataloader_params = {'batch_size': BATCH_SIZE, 'shuffle': True}

    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    test_dataloader = DataLoader(test_dataset, **dataloader_params)

    sgd_optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    average_test_loss = 0
    average_train_loss = 0

    for i in range(NUM_EPOCHS):
        print(f"Epoch {i}")
        # Training mode
        model.train()
        total_train_loss = 0
        for batch_num, (train_positions, train_valid_moves, train_values) in enumerate(train_dataloader):
            sgd_optimizer.zero_grad()

            move_logits, value = model(train_positions)
            valid_move_probs = train_valid_moves / \
                train_valid_moves.sum(dim=1, keepdims=True)
            loss = F.kl_div(move_logits, valid_move_probs,
                            reduction='batchmean') + F.mse_loss(value, train_values.unsqueeze(dim=1))

            loss.backward()
            sgd_optimizer.step()

            total_train_loss += loss.item()
            if batch_num % 10 == 0:
                print(f"{batch_num+1:3d}, Loss: {loss.item():.4f}")

    # Test Evaluation
    model.eval()

    total_test_loss = 0
    with torch.no_grad():
        for test_positions, test_valid_moves, test_values in test_dataloader:
            test_move_logits, test_pred_values = model(test_positions)
            valid_move_probs = test_valid_moves / \
                test_valid_moves.sum(dim=1, keepdims=True)
            test_loss = (
                F.kl_div(test_move_logits, valid_move_probs,
                         reduction='batchmean')
                + F.mse_loss(test_pred_values, test_values.unsqueeze(dim=1)))
            total_test_loss += test_loss.item()

    average_train_loss = total_train_loss / len(train_dataloader)
    average_test_loss = total_test_loss / len(test_dataloader)
    loss_observer.record([average_train_loss, average_test_loss])
    print(f"Average Test Loss: {average_test_loss:.4f}")

    results = {'final_train_loss': average_train_loss,
               'final_test_loss': average_test_loss}

    loss_observer.plot()

    return results
