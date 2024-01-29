import chess
import random

from action import action_to_move
from mcts import MCTSNode, mcts_choose_move, expand_node
from agent import MCTSAgent, RandomChessAgent

import sys

MATE_IN_ONE = '1k6/6R1/1K6/8/8/8/8/8 w - - 2 2'
MATE_IN_TWO = 'k7/6R1/2K5/8/8/8/8/8 w - - 0 1'


def main():
    total_wins = 0
    total_draws = 0
    total_losses = 0
    for i in range(100):
        print(f"Game {i}", end="\t")
        sys.stdout.flush()
        moves_played, mcts_win, draw, mcts_lose = play_agent_vs_agent()
        total_wins += mcts_win
        total_draws += draw
        total_losses += mcts_lose
        print(f"Moves {moves_played:3d} | MCTS Win / Draw / Lose:"
              f" {mcts_win:2d} {draw:2d} {mcts_lose:2d} | Total"
              f" {total_wins:3d} / {total_draws:3d} / {total_losses:3d}")
        sys.stdout.flush()

    print(f"Total MCTS win / draw / lose "
          f"{total_wins:2d} / {total_draws:2d} / {total_losses:2d}")


def play_agent_vs_agent():
    agent = MCTSAgent()
    other = RandomChessAgent()

    moves_played = 0
    board = chess.Board()
    r = random.randint(0, 1)  # which out of [agent, other] goes first
    white = [agent, other][r]
    black = [agent, other][1-r]

    while not board.is_game_over():
        current_agent = white if moves_played % 2 == 0 else black
        # print(f"white {white} black {black}")
        # print(f"{current_agent}'s move")
        move = current_agent.choose_move()
        tab = '\t...' if moves_played % 2 else ' '
        # print(f"{moves_played:3d}.{tab}{move}")
        board.push(move)
        # print(board)
        agent.update_position(move)
        other.update_position(move)
        moves_played += 1

    # When game is terminated
    if board.is_checkmate():
        who_won = white if board.turn == chess.BLACK else black
        if who_won == agent:
            return (moves_played, 1, 0, 0)
        else:
            return (moves_played, 0, 0, 1)

    # The game ended in draw
    return (moves_played, 0, 1, 0)


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


# recursively_print_node(root_node)
if __name__ == "__main__":
    main()
