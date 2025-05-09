import matplotlib.pyplot as plt
import numpy as np
import time
import os
from typing import List, Optional, Dict, Any
import pandas as pd
import seaborn as sns


class EnhancedVisualizer:
    """Advanced visualization tool for Snake AI performance monitoring"""

    def __init__(self, save_dir: str = './visualizations'):
        """
        Initialize the visualizer

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir

        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create subdirectories for different visualization types
        self.train_dir = os.path.join(save_dir, 'training')
        self.test_dir = os.path.join(save_dir, 'testing')
        self.analysis_dir = os.path.join(save_dir, 'analysis')

        for directory in [self.train_dir, self.test_dir, self.analysis_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Enable interactive plotting
        plt.ion()

        # Set the default style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')

        # Initialize time tracking
        self.start_time = time.time()

        # Initialize data storage
        self.training_data = {
            'scores': [],
            'mean_scores': [],
            'epsilon': [],
            'loss': [],
            'game_lengths': [],
            'rewards': [],
            'steps': [],
            'timestamps': []
        }

        self.testing_data = {
            'scores': [],
            'moves': [],
            'durations': [],
            'efficiency': [],
            'seeds': []
        }

    def log_training_step(self,
                          score: int,
                          mean_score: float,
                          epsilon: float,
                          loss: float = None,
                          game_length: int = None,
                          total_reward: float = None) -> None:
        """
        Log data from a training episode

        Args:
            score: Episode score
            mean_score: Running mean score
            epsilon: Current exploration rate
            loss: Training loss (optional)
            game_length: Number of moves in the game (optional)
            total_reward: Sum of rewards for the episode (optional)
        """
        self.training_data['scores'].append(score)
        self.training_data['mean_scores'].append(mean_score)
        self.training_data['epsilon'].append(epsilon)
        self.training_data['loss'].append(loss if loss is not None else np.nan)
        self.training_data['game_lengths'].append(
            game_length if game_length is not None else len(self.training_data['scores']))
        self.training_data['rewards'].append(total_reward if total_reward is not None else score * 10)
        self.training_data['steps'].append(len(self.training_data['scores']))
        self.training_data['timestamps'].append(time.time() - self.start_time)

    def log_test_result(self,
                        score: int,
                        moves: int,
                        duration: float,
                        efficiency: float,
                        seed: int) -> None:
        """
        Log data from a test game

        Args:
            score: Game score
            moves: Number of moves
            duration: Game duration in seconds
            efficiency: Score per move ratio
            seed: Random seed used for the game
        """
        self.testing_data['scores'].append(score)
        self.testing_data['moves'].append(moves)
        self.testing_data['durations'].append(duration)
        self.testing_data['efficiency'].append(efficiency)
        self.testing_data['seeds'].append(seed)

    def plot_training_progress(self,
                               show: bool = True,
                               save: bool = True,
                               episode_num: int = None) -> None:
        """
        Create a comprehensive plot of training progress

        Args:
            show: Whether to display the plot
            save: Whether to save the plot
            episode_num: Current episode number for filename
        """
        if len(self.training_data['scores']) == 0:
            print("No training data to plot")
            return

        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # Plot scores and mean scores
        ax1 = axs[0, 0]
        x = np.arange(len(self.training_data['scores']))
        ax1.plot(x, self.training_data['scores'], label='Score', color='blue', alpha=0.6)
        ax1.plot(x, self.training_data['mean_scores'], label='Mean Score', color='red', linewidth=2)
        ax1.set_title('Scores Over Time')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Score')
        ax1.grid(True)
        ax1.legend()

        # Plot epsilon decay
        ax2 = axs[0, 1]
        ax2.plot(x, self.training_data['epsilon'], color='green', label='Exploration Rate')
        ax2.set_title('Exploration Rate (Epsilon) Decay')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Epsilon')
        ax2.grid(True)

        # Plot loss if available
        ax3 = axs[1, 0]
        valid_losses = [l for l, v in zip(self.training_data['loss'], x) if not np.isnan(l)]
        valid_x = [v for l, v in zip(self.training_data['loss'], x) if not np.isnan(l)]

        if valid_losses:
            ax3.plot(valid_x, valid_losses, color='purple', label='Training Loss')
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'No loss data available', horizontalalignment='center',
                     verticalalignment='center', transform=ax3.transAxes)

        # Plot game length (moves per episode)
        ax4 = axs[1, 1]
        ax4.plot(x, self.training_data['game_lengths'], color='orange', label='Game Length')
        ax4.set_title('Game Length (Moves per Episode)')
        ax4.set_xlabel('Episodes')
        ax4.set_ylabel('Moves')
        ax4.grid(True)

        # Add timestamp and episode info
        elapsed_time = time.time() - self.start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        episode_info = f"Episode: {episode_num}" if episode_num is not None else f"Episodes: {len(self.training_data['scores'])}"

        fig.text(0.5, 0.01, f"Training Time: {time_str} - {episode_info}",
                 ha='center', fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        if save:
            episode_suffix = f"_episode_{episode_num}" if episode_num is not None else ""
            filename = os.path.join(self.train_dir, f"training_progress{episode_suffix}.png")
            plt.savefig(filename)
            print(f"Training plot saved to {filename}")

        # Show the figure
        if show:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.close(fig)

    def plot_test_results(self,
                          show: bool = True,
                          save: bool = True) -> None:
        """
        Create visualizations for test results

        Args:
            show: Whether to display the plots
            save: Whether to save the plots
        """
        if len(self.testing_data['scores']) == 0:
            print("No test data to plot")
            return

        # Convert test data to DataFrame for easier manipulation
        test_df = pd.DataFrame(self.testing_data)

        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Test Results Analysis', fontsize=16)

        # Plot scores
        ax1 = axs[0, 0]
        sns.barplot(x=test_df.index, y='scores', data=test_df, ax=ax1)
        ax1.set_title('Scores by Test Game')
        ax1.set_xlabel('Test Game')
        ax1.set_ylabel('Score')

        # Add score values on bars
        for i, score in enumerate(test_df['scores']):
            ax1.text(i, score / 2, str(score), ha='center', va='center',
                     color='white', fontweight='bold')

        # Plot moves per game
        ax2 = axs[0, 1]
        sns.barplot(x=test_df.index, y='moves', data=test_df, ax=ax2)
        ax2.set_title('Moves by Test Game')
        ax2.set_xlabel('Test Game')
        ax2.set_ylabel('Moves')

        # Plot efficiency (score/move)
        ax3 = axs[1, 0]
        sns.barplot(x=test_df.index, y='efficiency', data=test_df, ax=ax3)
        ax3.set_title('Efficiency (Score/Move) by Test Game')
        ax3.set_xlabel('Test Game')
        ax3.set_ylabel('Efficiency')

        # Plot duration
        ax4 = axs[1, 1]
        sns.barplot(x=test_df.index, y='durations', data=test_df, ax=ax4)
        ax4.set_title('Game Duration by Test Game')
        ax4.set_xlabel('Test Game')
        ax4.set_ylabel('Duration (seconds)')

        # Add summary statistics
        summary = f"Average Score: {test_df['scores'].mean():.2f} | " \
                  f"Max Score: {test_df['scores'].max()} | " \
                  f"Average Efficiency: {test_df['efficiency'].mean():.4f}"

        fig.text(0.5, 0.01, summary, ha='center', fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        if save:
            filename = os.path.join(self.test_dir, "test_results.png")
            plt.savefig(filename)
            print(f"Test results plot saved to {filename}")

        # Show the figure
        if show:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.close(fig)

    def create_comparison_plots(self, show: bool = True, save: bool = True) -> None:
        """
        Create plots comparing training and testing performance

        Args:
            show: Whether to display the plots
            save: Whether to save the plots
        """
        if len(self.training_data['scores']) == 0 or len(self.testing_data['scores']) == 0:
            print("Insufficient data for comparison plots")
            return

        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Training vs Testing Performance', fontsize=16)

        # Plot score distributions
        ax1 = axs[0]

        # Create histograms for training and test scores
        bins = np.linspace(0, max(max(self.training_data['scores']),
                                  max(self.testing_data['scores'])) + 5, 20)

        ax1.hist(self.training_data['scores'], bins=bins, alpha=0.7,
                 label='Training Scores', color='blue')
        ax1.hist(self.testing_data['scores'], bins=bins, alpha=0.7,
                 label='Test Scores', color='red')

        ax1.set_title('Score Distribution Comparison')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # Calculate and plot efficiency (score per move)
        ax2 = axs[1]

        # For training data, efficiency is approximated as score/game_length
        training_efficiency = [s / max(1, gl) for s, gl in
                               zip(self.training_data['scores'],
                                   self.training_data['game_lengths'])]

        # Create boxplot comparison
        box_data = [training_efficiency, self.testing_data['efficiency']]
        box_labels = ['Training', 'Testing']

        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_title('Efficiency Comparison (Score per Move)')
        ax2.set_ylabel('Efficiency')

        # Add summary statistics
        training_mean = np.mean(self.training_data['scores'])
        testing_mean = np.mean(self.testing_data['scores'])

        summary = f"Training Mean Score: {training_mean:.2f} | " \
                  f"Testing Mean Score: {testing_mean:.2f} | " \
                  f"Improvement: {100 * (testing_mean - training_mean) / max(1, training_mean):.2f}%"

        fig.text(0.5, 0.01, summary, ha='center', fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        if save:
            filename = os.path.join(self.analysis_dir, "training_vs_testing.png")
            plt.savefig(filename)
            print(f"Comparison plot saved to {filename}")

        # Show the figure
        if show:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.close(fig)

    def create_learning_curve_analysis(self,
                                       window_size: int = 50,
                                       show: bool = True,
                                       save: bool = True) -> None:
        """
        Create an analysis of the learning curve with trend lines

        Args:
            window_size: Window size for rolling averages
            show: Whether to display the plots
            save: Whether to save the plots
        """
        if len(self.training_data['scores']) < window_size:
            print(f"Not enough training data for learning curve analysis (need at least {window_size} episodes)")
            return

        # Convert to pandas for easier rolling calculations
        train_df = pd.DataFrame(self.training_data)

        # Calculate rolling statistics
        train_df['rolling_score'] = train_df['scores'].rolling(window=window_size).mean()
        train_df['rolling_std'] = train_df['scores'].rolling(window=window_size).std()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot raw scores with transparency
        ax.plot(train_df['steps'], train_df['scores'], 'o', markersize=3,
                alpha=0.3, color='blue', label='Individual Scores')

        # Plot rolling mean
        ax.plot(train_df['steps'], train_df['rolling_score'],
                linewidth=2.5, color='red', label=f'Rolling Mean (window={window_size})')

        # Add confidence intervals (mean ± std)
        ax.fill_between(
            train_df['steps'],
            train_df['rolling_score'] - train_df['rolling_std'],
            train_df['rolling_score'] + train_df['rolling_std'],
            color='red', alpha=0.2, label='±1 Std Dev'
        )

        # Try to fit a polynomial curve to show learning trend
        valid_indices = ~np.isnan(train_df['rolling_score'])
        if sum(valid_indices) > 3:
            x = train_df['steps'][valid_indices]
            y = train_df['rolling_score'][valid_indices]

            # Fit 2nd degree polynomial
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)

            # Plot trend line
            ax.plot(x, p(x), '--', linewidth=2, color='green',
                    label='Learning Trend (Polynomial Fit)')

        # Set title and labels
        ax.set_title('Learning Curve Analysis', fontsize=14)
        ax.set_xlabel('Training Episode', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Add annotations for final performance
        if len(train_df) > 0:
            final_mean = train_df['rolling_score'].iloc[-1]
            if not np.isnan(final_mean):
                ax.annotate(f'Final Mean: {final_mean:.2f}',
                            xy=(train_df['steps'].iloc[-1], final_mean),
                            xytext=(10, 30), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'))

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        if save:
            filename = os.path.join(self.analysis_dir, "learning_curve_analysis.png")
            plt.savefig(filename)
            print(f"Learning curve analysis saved to {filename}")

        # Show the figure
        if show:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.close(fig)

    def save_data_to_csv(self) -> None:
        """Save all collected data to CSV files for external analysis"""
        # Save training data
        if self.training_data['scores']:
            train_df = pd.DataFrame(self.training_data)
            train_file = os.path.join(self.save_dir, 'training_data.csv')
            train_df.to_csv(train_file, index=False)
            print(f"Training data saved to {train_file}")

        # Save testing data
        if self.testing_data['scores']:
            test_df = pd.DataFrame(self.testing_data)
            test_file = os.path.join(self.save_dir, 'testing_data.csv')
            test_df.to_csv(test_file, index=False)
            print(f"Testing data saved to {test_file}")

    def generate_performance_report(self) -> None:
        """Generate comprehensive performance visualizations and reports"""
        # Create all plots
        if self.training_data['scores']:
            self.plot_training_progress(show=False, save=True)
            self.create_learning_curve_analysis(show=False, save=True)

        if self.testing_data['scores']:
            self.plot_test_results(show=False, save=True)

        if self.training_data['scores'] and self.testing_data['scores']:
            self.create_comparison_plots(show=False, save=True)

        # Save data
        self.save_data_to_csv()

        print(f"\nPerformance report generated in {self.save_dir}")


def save_training_data(scores: List[int],
                       mean_scores: List[float],
                       filename: str = 'training_data.npz') -> None:
    """
    Save training data to file

    Args:
        scores: List of game scores
        mean_scores: List of mean scores
        filename: File to save data to
    """
    np.savez(filename, scores=np.array(scores), mean_scores=np.array(mean_scores))
    print(f"Training data saved to {filename}")


def load_training_data(filename: str = 'training_data.npz') -> tuple:
    """
    Load training data from file

    Args:
        filename: File to load data from

    Returns:
        Tuple of (scores, mean_scores)
    """
    if os.path.exists(filename):
        data = np.load(filename)
        scores = data['scores'].tolist()
        mean_scores = data['mean_scores'].tolist()
        print(f"Training data loaded from {filename}")
        return scores, mean_scores
    else:
        print(f"No training data found at {filename}")
        return [], []