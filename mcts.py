# MCTS Algorithm
# With no neural net (for now)

import chess
import numpy as np

from typing import List, Tuple, Dict
from action import move_to_action, action_to_move

NUM_ACTIONS = 64 * 64

N_SIM = 100
C_PUCT = 1

SAMPLING_TEMPERATURE = 4


class MCTSNode:
    """
    One node in the Monte carlo search tree. Represents one specific board
    state.
    """

    def __init__(self):
        self.q: np.ndarray = np.zeros(NUM_ACTIONS)  # average evaluation
        self.w: np.ndarray = np.zeros(NUM_ACTIONS)  # sum of q values
        self.n: np.ndarray = np.zeros(NUM_ACTIONS)  # number of visits
        self.p: np.ndarray = np.zeros(NUM_ACTIONS)  # prior probabilities
        # mask for legal moves (will be reused every time we go thru this node)
        self.legal_mask: np.ndarray = np.zeros(NUM_ACTIONS)
        # should always be equal to self.n.sum(), storing here for memoizing
        self.n_sum = 0
        # one node for every expanded action
        self.next_states: Dict[int, MCTSNode] = {}


def mcts_choose_move(root_node: MCTSNode, board: chess.Board) -> Tuple[chess.Move, np.ndarray, float]:
    """
    Performs Monte Carlo Tree Search over the action space, guided by the neural
    network.

    This function returns not only one chosen move but also a new set of
    probabilities associated with the action space and also a new average
    evaluation.

    It takes an argument for the root node so we can continue the search from an
    already existing tree for playing next moves after the first one. The board
    argument is required because the nodes themselves do not store the current
    board state. This is because (1) storing the board state would require lots
    of copy operations which would be slow and (2) because only the board state
    of the root node would end up in the training data. Therefore it takes the
    board state and makes/unmakes moves on it. The board state after the search
    should be same as before the search.

    It performs N_SIM simulations from the root node (board) where each
    simulation starts at the root node and selects a path through the tree based
    on the probabilities and evaluations of moves/actions and ultimately reaches
    any one leaf node. The leaf node is then evaluated by the neural network and
    the new probabilities and evaluation is backed up and updated throughout the
    path. Note that the probabilities and evaluations are stored for each _edge_
    (i.e. move/action) in the tree, and not for the node itself.

    The evaluation of the current position is supposed to be between
    +1 (winning) or -1 (losing). It is from the perspective of the current
    player.

    So, for example, when one of the player gets checkmated, the evaluation
    from their perspective is always (-1).

    It will choose the next node to expand based on a score (S) which depends on
    the prior probability (P) of the node and the average evaluation (Q). It
    will also increase the probability of visiting a node if it has been
    explored relatively fewer times (N) (to ensure sufficient exploration).

        S_a = Q + c * P * sqrt(sum(N)) / (1 + N_a)

    The action a with highest score S_a will be selected. At the end of the
    simulation it will expand the new node by calling the neural_net_eval
    function on it, and then back up the search tree updating the visit counts
    and mean values of Q.
    """

    # 1. Start from the root node
    # (For making sure the board state is not changed after the search.)
    # saved_start_fen = board.fen()

    # 2. Run N_sim simulations
    for i in range(N_SIM):
        # print(f"Simulation {i}", end=" ")
        simulate(root_node, board)
        # print()

    # 3. Sample chosen move and report new probs and eval.
    # Sample move from root probabilities proportional to visit counts.
    # (todo: add temperature parameter, and reduce temperature to 0 towards the
    # end of the game.)
    root_probs = root_node.n.copy()
    root_probs **= SAMPLING_TEMPERATURE
    root_probs /= root_probs.sum()
    action = np.random.choice(np.arange(0, NUM_ACTIONS), p=root_probs)
    move = action_to_move(action, board)
    root_eval = root_node.w.sum() / root_node.n_sum

    # (make sure we didn't accidentally modify the board)
    # assert board.fen() == saved_start_fen

    # Return chosen move, new probabilities and new evaluation for this state
    return (move, root_probs, root_eval)


def simulate(root_node: MCTSNode, board: chess.Board) -> None:
    """
    Performs one simulation from the root node in an MCTS search.
    """
    # (For making sure the board state is not changed after the simulation.)
    # saved_start_fen = board.fen()

    # 1. Select state to expand until we reach a leaf node.

    current_node = root_node
    path: List[Tuple[MCTSNode, int]] = []
    chosen_action: int | None = None
    while True:
        # select node to expand
        action = select_action(current_node)
        move = action_to_move(action, board)
        # print(f"{move}", end=" ")
        board.push(move)
        path.append((current_node, action))
        if action in current_node.next_states:
            # did not reach a leaf node yet
            current_node = current_node.next_states[action]
        else:
            # reached a leaf node to expand
            chosen_action = action
            break

    # 2. Expand state when we reach a leaf node.

    # Note: only the following two steps actually use the `board` state which
    # might be slow.

    # check current board state for termination
    if board.is_game_over() or board.can_claim_fifty_moves():
        value = expand_terminal_node(current_node, board)
    else:
        # Create a new node and assign the policy to it
        # And attach it to the current node
        new_node = MCTSNode()
        value = expand_node(new_node, board)
        current_node.next_states[chosen_action] = new_node

    # 3. Back up new evaluation and visit count values.
    # Note: value will change sign for each player
    # first time, value will be - as it is added to its parent node
    value_sign = -1
    for node, action in reversed(path):
        node.w[action] += value_sign * value
        node.n[action] += 1
        node.n_sum += 1
        node.q[action] = node.w[action] / node.n[action]
        board.pop()
        value_sign *= -1

    # (make sure we didn't accidentally modify the board)
    # assert board.fen() == saved_start_fen
    return


def expand_node(node: MCTSNode, board: chess.Board) -> float:
    policy, value = evaluate_board(board)
    legal_mask = get_legal_mask(board)
    node.p = policy
    node.legal_mask = legal_mask
    return value


def expand_terminal_node(node: MCTSNode, board: chess.Board) -> float:
    # value is always w.r.t. the current player in the board state, so
    # If a player gets checkmated then the value is always -1. The only
    # outcomes of a game are (checkmate, draw), and in case of a draw the
    # value is 0.
    value = -1 if board.is_checkmate() else 0
    return value


def get_legal_mask(board: chess.Board) -> np.ndarray:
    # generate a legal moves mask
    legal_mask = np.zeros(NUM_ACTIONS)
    for move in board.legal_moves:
        legal_mask[move_to_action(move)] = 1
    return legal_mask


def evaluate_board(board: chess.Board) -> Tuple[np.ndarray, float]:
    # Note: the following neural net evaluation step will be slow. We will
    # need to implement a queue and batched processing for it later.
    policy, value = neural_net_eval(board)
    return policy, value


def select_action(node: MCTSNode) -> int:
    """
    Returns the action with the highest UCT Score.
    """
    # calculate uct scores
    s: np.ndarray
    s = node.q + C_PUCT * node.p * np.sqrt(node.n_sum) * 1 / (1 + node.n)
    # apply legal moves mask before getting argmax
    # assert s.min() >= -1
    s += 1  # making sure there are no negative elements
    s *= node.legal_mask
    return np.argmax(s).item()


def neural_net_eval(board: chess.Board) -> Tuple[np.ndarray, float]:
    """
    This function is an interface to a neural net that returns two things,
    a policy and an evaluation (between +1 and -1) for the given position.
    This mock implementation returns a uniform distribution and always 0 for the
    evaluation.
    """
    return np.ones(NUM_ACTIONS), 0
