# Snake AI: Reinforcement Learning Project

## 📋 Overview

This project implements a reinforcement learning agent that learns to play the classic Snake game using Deep Q-Learning (DQN). The AI agent uses a neural network to predict optimal moves based on the game state and improves through experience.

## 🚀 Features

- **Deep Q-Learning Implementation**: Uses PyTorch to build and train a neural network for Q-learning
- **Training & Testing Modes**: Separate modes for training new models and testing existing ones
- **Comprehensive Visualization**: Real-time visualization of learning progress with performance metrics
- **Customizable Parameters**: Easily adjust learning parameters, network architecture, and game settings
- **Performance Analytics**: Track scores, efficiency, loss curves, and other key metrics
- **Model Saving & Loading**: Continue training from previously saved models
- **Deterministic Testing**: Test model performance using fixed random seeds for reproducibility

## 🛠️ Installation

Clone the repository and install the required packages:

### Requirements

```
numpy==2.2.4
gym==0.26.2
pygame==2.6.1
torch==2.6.0
matplotlib==3.10.1
Ipython==9.2.0
pandas==2.2.3
seaborn==0.13.2
imageio==2.37.0
pillow==11.2.1
```

## 🎮 Usage

### Training a Model

To train a new model:

```bash
python snake_main.py --train --episodes 500 --plot
```

### Continue Training from Existing Model

```bash
python snake_main.py --train --continue_training --model_file model_final.pth --episodes 200
```

### Testing a Trained Model

```bash
python snake_main.py --test --model_file ./model/model_final.pth --games 10
```

### Available Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train` | Run in training mode | - |
| `--test` | Run in testing mode | - |
| `--speed` | Game speed | 40 |
| `--render` | Rendering mode ('rgb_array' or 'human') | 'rgb_array' |
| `--hidden_size` | Size of hidden layer in neural network | 256 |
| `--model_file` | Model file name | 'model_final.pth' |
| `--plot` | Show performance plots interactively | False |
| `--episodes` | Max training episodes | 1000 |
| `--lr` | Learning rate | 0.001 |
| `--gamma` | Discount factor | 0.9 |
| `--save_every` | Save model every N episodes | 100 |
| `--plot_every` | Update plot every N episodes during training | 10 |
| `--save_data` | Save training data to .npz file | False |
| `--continue_training` | Continue training from existing model | False |
| `--games` | Number of games to test | 5 |
| `--seeds` | Comma-separated random seeds for testing | '42,123,256,789,1024' |
| `--no_vis` | Disable human visualization for testing | False |

## 🧠 How It Works

### Deep Q-Learning

The agent learns to play Snake using Deep Q-Learning:

1. **State Representation**: The game state is encoded as an 11-element vector containing:
   - Danger detection in straight, right, and left directions
   - Current direction of the snake (one-hot encoded)
   - Relative position of food to snake head (binary)

2. **Neural Network Architecture**:
   - Input layer: 11 neurons (state size)
   - Hidden layer: 256 neurons (configurable)
   - Output layer: 3 neurons (straight, right turn, left turn)
   - Activation function: ReLU

3. **Exploration vs. Exploitation**:
   - Uses an epsilon-greedy policy that decreases over time
   - Starting with high randomness that gradually shifts to learned strategy

4. **Replay Memory**:
   - Stores experiences as (state, action, reward, next_state, done) tuples
   - Enables mini-batch learning from past experiences

5. **Reward System**:
   - Positive reward for eating food
   - Negative reward for game over
   - Small negative reward for each step to encourage efficient paths

### Visualization

The project includes comprehensive visualization tools:
- Real-time training progress charts
- Performance metrics tracking
- Saved visualizations for later analysis

## 📊 Project Structure

```
snake-ai/
├── snake_main.py         # Main script for training and testing
├── snake_game.py         # Snake game implementation for AI
├── snake_game_human.py   # Snake game implementation for human players
├── visualization.py      # Visualization and data logging utilities
├── requirements.txt      # Required Python packages
├── model/                # Saved model files
└── visualizations/       # Training and testing visualizations
```

## 📈 Results

After training, the best model is saved in the model folder. The visualization directory contains detailed performance analysis for each training and testing run.

