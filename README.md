# AlphaZero
 This project implements an AlphaZero-based agent to play Tic-Tac-Toe using reinforcement learning. The agent learns through self-play, improving its strategy as it plays and trains. Three training configurations are available, each adding progressively more complexity to optimize training performance.

## Table of Contents
- [Video Demo](#video-demo)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
  - [AlphaZero](#alphazero)
  - [AlphaZero_Multi](#alphazero_multi)
  - [AlphaZero_Multi2](#alphazero_multi2)
- [Evaluating the Model](#evaluating-the-model)
- [Files and Directory Structure](#files-and-directory-structure)
- [Notes](#notes)

## Video Demo

https://github.com/user-attachments/assets/32d37897-4dde-44cc-af08-17fedbf168d6


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/alphazero-tictactoe.git
   cd alphazero-tictactoe

2. Install Dependencies: Make sure you have the required packages installed. You may need the following dependencies:
```bash
pip install numpy torch
```
3. Set Up Godot (for Evaluation): Download and install Godot Engine, if not already installed. Open the project in Godot to use the TicTacToe scene for evaluation.

## Getting Started

The project offers three versions for training, each utilizing a different configuration to balance performance and training speed. The training scripts and configurations are outlined below.

## Training the Model

### AlphaZero
The alphazero.py script is a CPU-only version of the training process. Use this if GPU resources are limited, or you want a baseline implementation.

To start training:
```bash
python alphazero.py
```

### AlphaZero_Multi
The alphazero_multi.py script enables multithreading to gather training samples more efficiently. Training occurs on the GPU, if available, to speed up the process.

To start multithreaded training:
```bash
python alphazero_multi.py
```

### AlphaZero_Multi2
The alphazero_multi2.py script builds on alphazero_multi.py by adding resumability, allowing you to save and resume training sessions.

To start multithreaded training with resumability:
```bash
python alphazero_multi2.py
```

## Training Notes
During training, self-play games are used to generate training data, which the model then uses to improve its performance. You can monitor training metrics, if implemented, to observe model progress and performance improvements.

## Evaluating the Model

Once training is complete, evaluate the trained model in a live Tic-Tac-Toe environment created in Godot.

Run the play.py Script to Start the Server:
```bash
python play.py
```
This will start a local server that the Godot client can connect to for game evaluation.
Open Godot and Load the TicTacToe Scene:
Open the project in Godot.
Navigate to and run the TicTacToe scene to play against or watch the model in action.
Observe Model Performance: Once connected, the model will play Tic-Tac-Toe, against the user, based on the strategy it learned during training.

## Files and Directory Structure

**alphazero.py**: CPU-only version for training the model.

**alphazero_multi.py**: Multithreaded version that uses the CPU for data gathering and GPU for training.

**alphazero_multi2.py**: Multithreaded version with resumability, enabling saved progress for training.

**play.py**: Script for starting the server for model evaluation.

**Godot Project Folder**: Contains the Godot project, including the TicTacToe scene for model evaluation.

## Notes

**Training Resumability**: The alphazero_multi2.py script allows you to save and load training progress, which can be particularly useful for longer training sessions.

**GPU Acceleration**: For faster training, use alphazero_multi.py or alphazero_multi2.py, as both support GPU training.

**Evaluation**: Ensure that play.py is running before starting the TicTacToe scene in Godot to avoid connection issues.
