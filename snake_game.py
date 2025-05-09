import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from typing import Tuple, List, Optional, Union

# Initialize pygame
pygame.init()

# Define a named tuple for points on the grid with x and y coordinates
Point = namedtuple('Point', 'x, y')


# Configuration constants
class GameConfig:
    # Display settings
    WIDTH = 640
    HEIGHT = 480
    BLOCK_SIZE = 20
    SPEED = 40

    # Colors (RGB format)
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)

    # Game parameters
    MAX_ITERATIONS_PER_FOOD = 100  # Maximum frames without eating before game over
    FONT_NAME = 'arial.ttf'
    FONT_SIZE = 25

    # Render mode options
    RENDER_MODE_HUMAN = 'human'
    RENDER_MODE_RGB_ARRAY = 'rgb_array'


# Direction enumeration
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGameAI:
    """Snake game environment for reinforcement learning"""

    def __init__(self, width=GameConfig.WIDTH, height=GameConfig.HEIGHT, render_mode=GameConfig.RENDER_MODE_HUMAN):
        """Initialize the game with given dimensions and render mode"""
        self.w = width
        self.h = height
        self.render_mode = render_mode

        # Initialize pygame display
        if self.render_mode == GameConfig.RENDER_MODE_HUMAN:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
        else:
            # Create a surface for rgb_array mode
            self.display = pygame.Surface((self.w, self.h))

        self.clock = pygame.time.Clock()

        # Load font for rendering text
        try:
            self.font = pygame.font.Font(GameConfig.FONT_NAME, GameConfig.FONT_SIZE)
        except pygame.error:
            # Fallback to default font if arial.ttf is not available
            self.font = pygame.font.SysFont('arial', GameConfig.FONT_SIZE)

        # Initialize game state
        self.reset()

    def reset(self) -> None:
        """Reset the game to initial state"""
        # Initialize snake direction and position
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)

        # Create snake body
        self.snake = [
            self.head,
            Point(self.head.x - GameConfig.BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * GameConfig.BLOCK_SIZE), self.head.y)
        ]

        # Reset score and place initial food
        self.score = 0
        self.food = None
        self._place_food()

        # Reset frame counter for timeout detection
        self.frame_iteration = 0

    def _place_food(self) -> None:
        """Place food at a random location not occupied by the snake"""
        x = random.randint(0, (self.w - GameConfig.BLOCK_SIZE) // GameConfig.BLOCK_SIZE) * GameConfig.BLOCK_SIZE
        y = random.randint(0, (self.h - GameConfig.BLOCK_SIZE) // GameConfig.BLOCK_SIZE) * GameConfig.BLOCK_SIZE
        self.food = Point(x, y)

        # If food spawns on snake, try again
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action: List[int]) -> Tuple[int, bool, int]:
        """
        Execute one time step of the game

        Args:
            action: One-hot encoded action [straight, right, left]

        Returns:
            reward: Reward for the current step
            game_over: Whether the game has ended
            score: Current score
        """
        self.frame_iteration += 1

        # Process events (allow quitting the game)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake based on action
        self._move(action)
        self.snake.insert(0, self.head)

        # Initialize reward
        reward = 0

        # Check for game over conditions
        game_over = False

        # Check for collision or timeout
        if self.is_collision() or self.frame_iteration > GameConfig.MAX_ITERATIONS_PER_FOOD * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if snake ate food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # Remove tail segment if no food eaten
            self.snake.pop()

        # Update display
        self._update_ui()

        # Control game speed
        if GameConfig.SPEED > 0:
            self.clock.tick(GameConfig.SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt: Optional[Point] = None) -> bool:
        """Check if there's a collision at the given point"""
        if pt is None:
            pt = self.head

        # Check boundary collision
        if (pt.x >= self.w or pt.x < 0 or
                pt.y >= self.h or pt.y < 0):
            return True

        # Check self-collision (snake biting itself)
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self) -> Union[None, np.ndarray]:
        """
        Update the game display

        Returns:
            np.ndarray if render_mode is rgb_array, otherwise None
        """
        # Fill background
        self.display.fill(GameConfig.BLACK)

        # Draw snake body
        for pt in self.snake:
            # Draw main body rectangle
            pygame.draw.rect(self.display, GameConfig.BLUE1,
                             pygame.Rect(pt.x, pt.y, GameConfig.BLOCK_SIZE, GameConfig.BLOCK_SIZE))
            # Draw inner rectangle for aesthetic
            pygame.draw.rect(self.display, GameConfig.BLUE2,
                             pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, GameConfig.RED,
                         pygame.Rect(self.food.x, self.food.y, GameConfig.BLOCK_SIZE, GameConfig.BLOCK_SIZE))

        # Draw score
        text = self.font.render(f"Score: {self.score}", True, GameConfig.WHITE)
        self.display.blit(text, [0, 0])

        # Update the display or return the pixels
        if self.render_mode == GameConfig.RENDER_MODE_HUMAN:
            pygame.display.flip()
            return None
        else:
            # Convert the pygame surface to a numpy array for rgb_array mode
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.display)),
                axes=(1, 0, 2)
            )

    def _move(self, action: List[int]) -> None:
        """
        Update the snake's direction and position based on action

        Args:
            action: One-hot encoded action [straight, right, left]
        """
        # Define clockwise direction order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Change direction based on action
        if np.array_equal(action, [1, 0, 0]):
            # Continue straight
            new_direction = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Turn right (clockwise)
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Turn left (counter-clockwise)
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx]

        # Update direction
        self.direction = new_direction

        # Update head position based on direction
        x, y = self.head.x, self.head.y

        if self.direction == Direction.RIGHT:
            x += GameConfig.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= GameConfig.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += GameConfig.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= GameConfig.BLOCK_SIZE

        # Create new head at updated position
        self.head = Point(x, y)

    def set_render_mode(self, render_mode: str) -> None:
        """
        Change the rendering mode during runtime

        Args:
            render_mode: Either 'human' or 'rgb_array'
        """
        if render_mode not in [GameConfig.RENDER_MODE_HUMAN, GameConfig.RENDER_MODE_RGB_ARRAY]:
            raise ValueError(
                f"Render mode must be one of {GameConfig.RENDER_MODE_HUMAN} or {GameConfig.RENDER_MODE_RGB_ARRAY}")

        # Only reinitialize display if changing modes
        if self.render_mode != render_mode:
            self.render_mode = render_mode

            if self.render_mode == GameConfig.RENDER_MODE_HUMAN:
                self.display = pygame.display.set_mode((self.w, self.h))
                pygame.display.set_caption('Snake AI')
            else:
                # Create a surface for rgb_array mode
                self.display = pygame.Surface((self.w, self.h))