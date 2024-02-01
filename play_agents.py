import chess
import random

from agent import ChessAgent


def play_agent_vs_agent(agent: ChessAgent, other: ChessAgent, verbose=False):
    moves_played = 0
    board = chess.Board()
    r = random.randint(0, 1)  # which out of [agent, other] goes first
    white = [agent, other][r]
    black = [agent, other][1-r]

    while not board.is_game_over():
        current_agent = white if moves_played % 2 == 0 else black
        # print(f"{current_agent}'s move")
        move = current_agent.choose_move()
        tab = '\t...' if moves_played % 2 else ' '
        board.push(move)
        if verbose:
            print(f"white {white} black {black}")
            print(f"{moves_played:3d}.{tab}{move}")
            print(board)
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
