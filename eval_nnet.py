from agent import NNetAgent, RandomChessAgent
from play_agents import play_agent_vs_agent

import sys
import agent

NUM_GAMES = 100
VERBOSE = False
agent.TEMPERATURE = 1


def main():
    total_wins = 0
    total_draws = 0
    total_losses = 0
    nnet_agent = NNetAgent()
    random_agent = RandomChessAgent()
    for i in range(NUM_GAMES):
        nnet_agent.reset()
        random_agent.reset()
        print(f"Game {i}", end="\t")
        sys.stdout.flush()
        moves_played, mcts_win, draw, mcts_lose = play_agent_vs_agent(
            nnet_agent, random_agent, verbose=VERBOSE)
        total_wins += mcts_win
        total_draws += draw
        total_losses += mcts_lose
        print(f"Moves {moves_played:3d} | NNET Win / Draw / Lose:"
              f" {mcts_win:2d} {draw:2d} {mcts_lose:2d} | Total"
              f" {total_wins:3d} / {total_draws:3d} / {total_losses:3d}")
        sys.stdout.flush()

    print(f"Total NNET win / draw / lose "
          f"{total_wins:2d} / {total_draws:2d} / {total_losses:2d}")


if __name__ == "__main__":
    main()