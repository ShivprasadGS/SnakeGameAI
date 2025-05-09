import random
import numpy as np
import argparse
import time
import os
from collections import deque
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Union, Tuple, List

from snake_game import SnakeGameAI, Direction, Point, GameConfig
from visualization import EnhancedVisualizer, save_training_data, load_training_data



class Config:
    # Training parameters
    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.001
    GAMMA = 0.9  # Discount rate
    EPSILON_START = 80  # Starting exploration rate
    EPSILON_DECAY = 0.5  # How quickly to reduce randomness
    EPSILON_MIN = 10  # Minimum exploration rate
    HIDDEN_LAYER_SIZE = 256
    MAX_EPISODES = 1000  # Maximum number of training episodes

    # Testing parameters
    TEST_SEEDS = [42, 123, 256, 789, 1024]
    TEST_GAMES = 5

    # General parameters
    RENDER_MODE = 'rgb_array'  # 'rgb_array' or 'human'
    SPEED = 40  # Game speed during training/testing
    MODEL_FILE = 'model_final.pth'  # Default model file name


class Linear_QNet(nn.Module):
    """Neural network for Q-learning, predicting action values"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_name: str = 'model.pth') -> None:
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            self.eval()
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}")


class QTrainer:
    """Trainer for the Q-network"""

    def __init__(self, model: Linear_QNet, lr: float, gamma: float):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Training on: {self.device}")

    def train_step(self,
                   state: Union[List, torch.Tensor],
                   action: Union[List, torch.Tensor],
                   reward: Union[List, torch.Tensor],
                   next_state: Union[List, torch.Tensor],
                   done: Union[List, Tuple]) -> float:
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                with torch.no_grad():
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Agent:
    def __init__(self, config=None):
        self.config = config or Config()
        self.n_games = 0
        self.epsilon = self.config.EPSILON_START
        self.gamma = self.config.GAMMA
        self.memory = deque(maxlen=self.config.MAX_MEMORY)
        self.model = Linear_QNet(
            input_size=11,
            hidden_size=self.config.HIDDEN_LAYER_SIZE,
            output_size=3
        )
        self.trainer = QTrainer(
            self.model,
            lr=self.config.LEARNING_RATE,
            gamma=self.gamma
        )

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - GameConfig.BLOCK_SIZE, head.y)
        point_r = Point(head.x + GameConfig.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - GameConfig.BLOCK_SIZE)
        point_d = Point(head.x, head.y + GameConfig.BLOCK_SIZE)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        state = [
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or \
            (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or \
            (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or \
            (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x, game.food.x > game.head.x,
            game.food.y < game.head.y, game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> Union[float, None]:
        """Train on a batch from replay memory. Returns loss from this training step."""
        if len(self.memory) == 0:
            return None

        if len(self.memory) > self.config.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.config.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done) -> float:
        """Train on a single experience tuple. Returns loss."""
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss

    def get_action(self, state, training=True) -> List[int]:
        final_move = [0, 0, 0]
        if training:
            self.epsilon = max(
                self.config.EPSILON_MIN,
                self.config.EPSILON_START - self.n_games * self.config.EPSILON_DECAY
            )
            if random.random() < self.epsilon / 200:  # Normalize epsilon effect
                move = random.randint(0, 2)
                final_move[move] = 1
                return final_move

        state_tensor = torch.tensor(state, dtype=torch.float, device=self.trainer.device)
        with torch.no_grad():
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def parse_args():
    parser = argparse.ArgumentParser(description='Snake AI - Train or Test')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train a new model')
    mode_group.add_argument('--test', action='store_true', help='Test an existing model')
    parser.add_argument('--speed', type=int, default=Config.SPEED,
                        help=f'Game speed (default: {Config.SPEED})')
    parser.add_argument('--render', type=str, default=Config.RENDER_MODE, choices=['rgb_array', 'human'],
                        help=f'Rendering mode (default: {Config.RENDER_MODE})')
    parser.add_argument('--hidden_size', type=int, default=Config.HIDDEN_LAYER_SIZE,
                        help=f'Size of hidden layer (default: {Config.HIDDEN_LAYER_SIZE})')
    parser.add_argument('--model_file', type=str, default=Config.MODEL_FILE,
                        help=f'Model file name (default: {Config.MODEL_FILE})')
    parser.add_argument('--plot', action='store_true', help='Show performance plots interactively')
    parser.add_argument('--episodes', type=int, default=Config.MAX_EPISODES,
                        help=f'Max training episodes (default: {Config.MAX_EPISODES})')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help=f'Learning rate (default: {Config.LEARNING_RATE})')
    parser.add_argument('--gamma', type=float, default=Config.GAMMA,
                        help=f'Discount factor (default: {Config.GAMMA})')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save model every N episodes (default: 100)')
    parser.add_argument('--plot_every', type=int, default=10,
                        help='Update plot every N episodes during training (default: 10)')
    parser.add_argument('--save_data', action='store_true',
                        help='Save legacy training data (scores, mean_scores) to .npz file')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue training from existing model and .npz data')
    parser.add_argument('--games', type=int, default=Config.TEST_GAMES,
                        help=f'Number of games to test (default: {Config.TEST_GAMES})')
    parser.add_argument('--seeds', type=str,
                        default=','.join(str(s) for s in Config.TEST_SEEDS),
                        help=f'Comma-separated random seeds (default: {",".join(str(s) for s in Config.TEST_SEEDS)})')
    parser.add_argument('--no_vis', action='store_true',
                        help='Disable human visualization for testing (runs headless)')
    return parser.parse_args()


def train(args):
    print("\n=== SNAKE AI TRAINING ===")
    # ... (print other args) ...

    config = Config()
    config.MAX_EPISODES = args.episodes
    config.RENDER_MODE = args.render
    config.SPEED = args.speed
    config.LEARNING_RATE = args.lr
    config.GAMMA = args.gamma
    config.HIDDEN_LAYER_SIZE = args.hidden_size
    config.MODEL_FILE = args.model_file
    GameConfig.SPEED = config.SPEED

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    visualizer = EnhancedVisualizer(save_dir=f'./visualizations/training_run_{run_timestamp}')

    # For legacy .npz saving and mean score calculation
    plot_scores_legacy = []
    plot_mean_scores_legacy = []
    total_score_sum_legacy = 0
    record = 0

    agent = Agent(config)
    initial_n_games_count = 0

    if args.continue_training:
        try:
            print(f"Loading model from {args.model_file} to continue training...")
            agent.model.load(args.model_file)

            if args.save_data:  # Only load legacy data if it was intended to be used
                previous_scores, previous_mean_scores = load_training_data()  # From helper.py
                if previous_scores:
                    plot_scores_legacy = previous_scores
                    plot_mean_scores_legacy = previous_mean_scores
                    total_score_sum_legacy = sum(plot_scores_legacy)
                    agent.n_games = len(plot_scores_legacy)
                    initial_n_games_count = agent.n_games
                    record = max(plot_scores_legacy) if plot_scores_legacy else 0
                    print(f"Resuming from game {agent.n_games}. Legacy data loaded.")
                # Note: EnhancedVisualizer starts fresh for plots unless it's modified to load its own CSVs.
        except Exception as e:
            print(f"Error loading model/previous data: {e}. Starting fresh training.")
            # Reset states
            plot_scores_legacy, plot_mean_scores_legacy, total_score_sum_legacy, agent.n_games, record = [], [], 0, 0, 0
            initial_n_games_count = 0

    game = SnakeGameAI(render_mode=args.render)

    training_start_time = time.time()
    print("\nTraining started...")

    # Loop for the specified number of new episodes
    for episode_idx in range(config.MAX_EPISODES):
        current_episode_num = initial_n_games_count + episode_idx + 1
        agent.n_games = current_episode_num  # Make sure agent's game count is up-to-date for epsilon decay

        episode_total_reward = 0
        episode_steps = 0
        episode_losses_short_term = []

        game.reset()
        current_episode_done = False
        current_episode_score = 0  # Score for this specific episode

        while not current_episode_done:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, training=True)
            reward_step, current_episode_done, current_episode_score = game.play_step(final_move)
            state_new = agent.get_state(game)

            loss_short = agent.train_short_memory(state_old, final_move, reward_step, state_new, current_episode_done)
            if loss_short is not None:
                episode_losses_short_term.append(loss_short)

            agent.remember(state_old, final_move, reward_step, state_new, current_episode_done)

            episode_total_reward += reward_step
            episode_steps += 1

        # End of an episode
        loss_long = agent.train_long_memory()

        logged_loss = None
        if loss_long is not None:
            logged_loss = loss_long
        elif episode_losses_short_term:
            logged_loss = mean(episode_losses_short_term)

        if current_episode_score > record:
            record = current_episode_score
            agent.model.save(args.model_file)  # Save best model

        if current_episode_num % args.save_every == 0:
            model_filename = f"model_ep{current_episode_num}.pth"
            agent.model.save(model_filename)

        # Update legacy scores and mean scores
        plot_scores_legacy.append(current_episode_score)
        total_score_sum_legacy += current_episode_score
        # Mean score over all games played (including continued ones)
        mean_score_val_legacy = total_score_sum_legacy / len(plot_scores_legacy)
        plot_mean_scores_legacy.append(mean_score_val_legacy)

        print(
            f'Game {current_episode_num}/{initial_n_games_count + config.MAX_EPISODES}, '
            f'Score: {current_episode_score}, Record: {record}, Epsilon: {agent.epsilon:.2f}, '
            f'Steps: {episode_steps}, Ep_Reward: {episode_total_reward:.2f}, '
            f'Loss: {f"{logged_loss:.4f}" if logged_loss is not None else "N/A"}'
        )

        visualizer.log_training_step(
            score=current_episode_score,
            mean_score=mean_score_val_legacy,  # Use the overall mean for consistency
            epsilon=agent.epsilon,
            loss=logged_loss,
            game_length=episode_steps,
            total_reward=episode_total_reward
        )

        if args.plot and current_episode_num % args.plot_every == 0:
            visualizer.plot_training_progress(show=True, save=True, episode_num=current_episode_num)

        if args.save_data and current_episode_num % args.save_every == 0:
            save_training_data(plot_scores_legacy, plot_mean_scores_legacy)

    # End of training loop
    agent.model.save(args.model_file)  # Save final model
    if args.save_data:
        save_training_data(plot_scores_legacy, plot_mean_scores_legacy)

    visualizer.generate_performance_report()  # Saves all plots and CSVs

    elapsed_time = time.time() - training_start_time
    num_trained_episodes = config.MAX_EPISODES
    print(f"\nTraining completed. Episodes trained in this session: {num_trained_episodes}.")
    print(f"Total episodes overall: {current_episode_num}")  # agent.n_games should be this
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Final record: {record}")

    if plot_scores_legacy:
        session_scores = plot_scores_legacy[initial_n_games_count:] if args.continue_training else plot_scores_legacy
        if session_scores:
            print(f"Average score of this session: {mean(session_scores):.2f}")
        print(f"Overall average score (all games): {mean(plot_scores_legacy):.2f}")

    if args.plot:  # Show final training plot if requested
        visualizer.plot_training_progress(show=True, save=False, episode_num=current_episode_num)
        visualizer.create_learning_curve_analysis(show=True, save=False)


def test(args):
    print("\n=== SNAKE AI TESTING ===")
    print(f"Testing model: {args.model_file}, Games: {args.games}, Seeds: {args.seeds}")
    # ... (print other args) ...

    seeds = [int(seed) for seed in args.seeds.split(',')]
    render_mode = 'rgb_array' if args.no_vis else 'human'

    config = Config()  # Use default config for agent structure
    config.HIDDEN_LAYER_SIZE = args.hidden_size  # Allow hidden size override
    # Epsilon is not used in testing for action selection (training=False)

    agent = Agent(config)
    try:
        agent.model.load(args.model_file)
    except Exception as e:
        print(f"Error loading model: {e}. Testing with untrained model.")

    game = SnakeGameAI(render_mode=render_mode)
    GameConfig.SPEED = args.speed

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    visualizer = EnhancedVisualizer(save_dir=f'./visualizations/testing_run_{run_timestamp}')

    all_scores_local = []  # For immediate summary

    print("\nStarting test games...")
    for i, seed_val in enumerate(seeds[:args.games]):
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        game.reset()
        game_over = False
        current_test_score = 0
        current_test_moves = 0
        test_start_time = time.time()

        print(f"\nGame {i + 1}/{args.games} (Seed: {seed_val}):")

        while not game_over:
            state = agent.get_state(game)
            final_move = agent.get_action(state, training=False)
            _, game_over, current_test_score = game.play_step(final_move)
            current_test_moves += 1

        test_duration = time.time() - test_start_time
        test_efficiency = current_test_score / max(1, current_test_moves)

        all_scores_local.append(current_test_score)

        visualizer.log_test_result(
            score=current_test_score,
            moves=current_test_moves,
            duration=test_duration,
            efficiency=test_efficiency,
            seed=seed_val
        )

        print(f"  Score: {current_test_score}, Moves: {current_test_moves}, "
              f"Duration: {test_duration:.2f}s, Efficiency: {test_efficiency:.4f}")

    visualizer.generate_performance_report()  # Saves plots and CSVs

    if all_scores_local:
        print("\nTest Summary (from local calculation):")
        print(
            f"Average Score: {mean(all_scores_local):.2f}, Max: {max(all_scores_local)}, Min: {min(all_scores_local)}")
        # Further stats can be taken from visualizer's saved CSV if needed.

    if args.plot and all_scores_local:
        print("Displaying final test results plot...")
        visualizer.plot_test_results(show=True, save=False)  # Show plot, don't re-save


if __name__ == '__main__':
    args = parse_args()
    if args.train:
        train(args)
    elif args.test:
        test(args)