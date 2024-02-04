import mcts
from agent import MCTSAgent, RandomChessAgent
from model import get_model
from play_agents import play_agent_vs_agent

import sys

NUM_GAMES = 20
VERBOSE = False
mcts.N_SIM = 100
mcts.SAMPLING_TEMPERATURE = 5


def main():
    total_wins = 0
    total_draws = 0
    total_losses = 0
    model = get_model()
    for i in range(NUM_GAMES):
        print(f"Game {i}", end="\t")
        sys.stdout.flush()
        mcts_agent = MCTSAgent(model)
        random_agent = RandomChessAgent()
        moves_played, mcts_win, draw, mcts_lose = play_agent_vs_agent(
            mcts_agent, random_agent, verbose=VERBOSE)
        total_wins += mcts_win
        total_draws += draw
        total_losses += mcts_lose
        print(f"Moves {moves_played:3d} | MCTS Win / Draw / Lose:"
              f" {mcts_win:2d} {draw:2d} {mcts_lose:2d} | Total"
              f" {total_wins:3d} / {total_draws:3d} / {total_losses:3d}")
        sys.stdout.flush()

    print(f"Total MCTS win / draw / lose "
          f"{total_wins:2d} / {total_draws:2d} / {total_losses:2d}")


if __name__ == "__main__":
    main()
