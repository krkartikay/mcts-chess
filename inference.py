import chess
import torch
import queue
import threading
import torch.nn.functional as F
from concurrent.futures import Future
from typing import Tuple
from convert import board_to_tensor

from model import ChessModel

request_queue = queue.Queue()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def start_inference_worker(model: ChessModel):
    """
    Starts the inference worker in a background thread.
    """
    model.eval()
    worker_thread = threading.Thread(
        target=inference_worker, args=[model])
    worker_thread.daemon = True
    worker_thread.start()


def neural_net_eval(board: chess.Board) -> Tuple[torch.Tensor, float]:
    future = Future()
    request_queue.put((future, board_to_tensor(board).to(device)))
    policy, value = future.result()
    return policy, value


def inference_worker(model: ChessModel):
    while True:
        batch_futures = []
        batch_tensors = []

        while len(batch_tensors) < 64:
            try:
                future, board = request_queue.get(
                    block=(len(batch_tensors) == 0))
                batch_futures.append(future)
                batch_tensors.append(board)
            except queue.Empty:
                break

        # print(";", len(batch_tensors))
        batch_tensor = torch.stack(batch_tensors, dim=0)
        with torch.no_grad():
            logits, values = model(batch_tensor)
            probs = F.softmax(logits, dim=1)

        for future, move_probs, value in zip(batch_futures, probs, values):
            future.set_result((move_probs.to('cpu'), value.item()))
