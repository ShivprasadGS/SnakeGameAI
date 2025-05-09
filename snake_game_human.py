import pygame   # import pygame library for game development
import random   # import random library to generate random numbers
from enum import Enum   # import Enum library to generate enumerations
from collections import namedtuple  # import namedtuple library to create simple classes

# initialize all pygame modules
pygame.init()

# set up font for rendering text
font = pygame.font.Font('arial.ttf', 25)

# define enumeration for the direction in which snake can move
class Direction(Enum):
    RIGHT = 1   # moving to the right
    LEFT = 2    # moving to the left
    UP = 3      # moving up
    DOWN = 4    # moving down

# define a named tuple for points on the grid, with x and y coordinates
Point = namedtuple('Point', 'x, y')

# define RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# set the size of each block in the game (snake segments and food)
BLOCK_SIZE = 20

# set the speed of the game
SPEED = 15

# define the snake game class.
class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w  # width of the game window
        self.h = h  # height of the game window

        # initialize the display window
        self.display = pygame.display.set_mode((self.w, self.h))
        # set the window title
        pygame.display.set_caption('Snake')
        # create clock object to control game speed
        self.clock = pygame.time.Clock()

        # initialize the starting direction of the snake
        self.direction = Direction.RIGHT

        # set up the initial position of the snake's head
        self.head = Point(self.w/2, self.h/2)

        # add head and two segments behind it to create snake's body
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]

        self.score = 0          # initialize the score
        self.food = None        # initialize the food position
        self._place_food()      # place first piece of food on the board

    def _place_food(self):
        # randomly place food on the board within the game boundaries
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)

        # if the food spawns on the snake, place it again
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:       # check if the player closes the window
                pygame.quit()
                quit()
            
            if event.type == pygame.KEYDOWN:    # check for key presses
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
        # 2. move the snake in the direction specified by user
        self._move(self.direction)      # update the head position
        self.snake.insert(0, self.head)     # add the new head position to the snake body

        # 3. check if game over
        game_over = False
        if self._is_collision():    # check if the snake has collided with itself or the boundary
            game_over = True
            return game_over, self.score    # end the game and return score

        # 4. check if the snake has eaten the food or just move forward
        if self.head == self.food:  #
            self.score += 1     # increase score
            self._place_food()  # place new food
        else:
            self.snake.pop()    # remove last segment if no food is eaten

        # 5. update the display and control the game speed
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        # hits boundary?
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself?
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)        # fill the display with black background

        for pt in self.snake:       # draw the snake on the screen
            # draw the main body of the snake
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # draw the inner square of the snake
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # draw the food on the screen
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # display the current score on the screen
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])     # position the score in the top left corner
        pygame.display.flip()       # update the full display surface on the screen

    def _move(self, direction):
        # update the position of the snake's head based on the direction
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)     # set the new head position

# main part of the code that combines everything and runs the game
if __name__ == '__main__':
    game = SnakeGame()      # create an instance of SnakeGame Class

    # game loop
    while True:
        game_over, score = game.play_step()     # execute a step in the game

        if game_over == True:       # end the game loop if the game is over
            break

    print('Final Score', score)     # print the final score

    pygame.quit()       # Quit the pygame instance
