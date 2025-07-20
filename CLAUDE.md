# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an implementation of AlphaZero-style Monte Carlo Tree Search (MCTS) for chess using PyTorch. The codebase combines neural network evaluation with MCTS search to create a chess AI that learns through self-play.

## Dependencies and Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

The project requires:
- `chess==1.10.0` for chess game logic
- `numpy>=1.22.0` for numerical operations  
- PyTorch (not in requirements.txt, install separately)

## Key Executables

All main scripts can be run with `python <script>.py`:

- `python train_model.py` - Train the neural network model
- `python gen_selfplay_games.py` - Generate self-play training data
- `python eval_mcts.py` - Evaluate MCTS agent vs random player
- `python eval_nnet.py` - Evaluate neural network directly
- `python gen_random_dataset.py` - Generate random position dataset

## Core Architecture

### Neural Network (`model.py`)
- `ChessModel`: Simple feedforward network with value and policy heads
- Input: 7×8×8 tensor representing board state
- Output: Move probabilities (64×64 action space) and position value
- Model saving/loading via `saved_model.pth`

### MCTS Implementation (`mcts.py`)
- `MCTSNode`: Tree search node with Q/W/N statistics and prior probabilities
- Multi-threaded search with configurable parameters:
  - `N_SIM`: Number of simulations (default 2000)
  - `C_PUCT`: Exploration constant (1.75)
  - `SAMPLING_TEMPERATURE`: Move selection temperature
- Uses virtual loss for thread safety during concurrent simulations

### Inference System (`inference.py`)
- Background worker thread for batched neural network evaluation
- Queue-based system for handling multiple evaluation requests
- Used by MCTS to get neural network predictions efficiently

### Move Representation (`action.py`)
- Maps between chess moves and 64×64 action tensor indices
- Handles promotion moves and action space conversion

### Data Pipeline
- `convert.py`: Board ↔ tensor conversion
- `training.py`: Model training with position/move/value data
- Self-play game generation creates training data

### Agents (`agent.py`)
- `MCTSAgent`: Uses MCTS + neural network for move selection
- `RandomChessAgent`: Baseline random player

## Model Training Workflow

1. Generate initial random dataset: `python gen_random_dataset.py`
2. Train initial model: `python train_model.py`
3. Generate self-play games: `python gen_selfplay_games.py`
4. Continue training with self-play data: `python train_model.py`
5. Evaluate performance: `python eval_mcts.py`

## Key Configuration

MCTS parameters are set as module-level variables in `mcts.py`:
- `N_SIM`: Search depth
- `C_PUCT`: Exploration vs exploitation balance
- `MAX_WORKERS`: Threading for parallel search

The model automatically uses CUDA if available, falls back to CPU.