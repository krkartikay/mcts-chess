import chess
import numpy as np

from action import action_to_move
from mcts import MCTSNode, mcts_choose_move, expand_node

board = chess.Board()

root_node = MCTSNode()
policy, value = expand_node(root_node, board)
root_node.p = policy

print(board)

move, new_probs, new_eval = mcts_choose_move(root_node, board)

print(f"Chosen Move: {move}")
print(f"New eval:    {new_eval}")
print(f"New probs: sum: {new_probs.sum()}\n{new_probs.reshape((64, 64))}")


def recursively_print_node(node: MCTSNode, prefix=""):
    print(f"{prefix}Node")
    print(f"{prefix}n: sum: {node.n.sum()} value: {node.n}")
    print(f"{prefix}p: sum: {node.p.sum()} value: {node.p}")
    print(f"{prefix}q: sum: {node.q.sum()} value: {node.q}")
    print(f"{prefix}w: sum: {node.w.sum()} value: {node.w}")
    print(f"{prefix}Number of next states: {len(node.next_states)}")
    for i, action in enumerate(node.next_states):
        print(
            f"{prefix}Child node {i+1:2d}: Move {action_to_move(action, None)}"
            f" [n ={node.n[action]:6.3f}] [p ={node.p[action]:6.3f}]"
            f" [q ={node.q[action]:6.3f}] [w ={node.w[action]:6.3f}]")
        recursively_print_node(node.next_states[action], prefix + "| ")


recursively_print_node(root_node)
