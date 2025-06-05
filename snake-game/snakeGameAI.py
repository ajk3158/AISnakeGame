import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Deep Q Learning Neural Network

# Set of symbolic names bound to unique values
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
    
    def __init__(self, w=400, h=400):
        # Initialize all needed game elements
        # Window size
        self.w = w
        self.h = h
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    # reset/reinitialize to continuously play game and learn
    def reset(self):
        # Init game state
        # Randomize initial snake state for better learning
        
        # Random starting direction
        # self.direction = Direction(random.randint(1, 4))
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        print(self.head)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.time_since_food = 0
        
    def place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()
    
    def distanceToFood(self, x, y):
        foodX = self.food[0]
        foodY = self.food[1]
        
        return np.linalg.norm(np.array([x, y]) - np.array([foodX, foodY]))
        # return math.sqrt((x - foodX)**2 + (y-foodY)**2)
        

    def play_step(self, action, reward, sbUsed=False):
        prev_reward = reward

        self.frame_iteration += 1
        self.time_since_food += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # User Input
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.key == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:
            #         self.direction = Direction.DOWN
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if sbUsed==True:
            time_up = False
        if self.is_collision() or self.time_since_food > 70 * len(self.snake):
            print("GAME OVER")
            game_over = True
            reward = -100
            if sbUsed==False:
                return reward, game_over, self.score
            elif sbUsed==True:
                time_up = True
                return (self.snake, self.food, reward, False, time_up, self.score)
            
        x1 = self.head.x
        y1 = self.head.y
        x2 = self.snake[1].x
        y2 = self.snake[1].y
        # TODO: MAYBE INTEGRATE PAST DISTANCE TO ENSURE THAT SNAKE FOLLOWS
        # 4. Snake ate food, closer to food, or further from food
        eaten_reward = 0
        if self.head == self.food:
            self.time_since_food = 0
            self.score += 1
            reward = 100
            eaten_reward = 10000
            self.place_food()
        elif self.distanceToFood(x1, y1) < self.distanceToFood(x2, y2):
            reward = 1 
            self.snake.pop()
        else:
            reward = -1
            self.snake.pop()
        
        
        # 5. update ui and clock
        self.update_ui()
        self.clock.tick(SPEED)
        # reward -= 0.1
        # 6. return game over and score
        if sbUsed==False:
            return reward, game_over, self.score
        elif sbUsed==True:
            # Diameter of square = 566
            reward = 10/((self.distanceToFood(x1, y1)) + eaten_reward) - prev_reward
            return (self.snake, self.food, reward, game_over, False, self.score)
    
    def is_collision(self, pt = None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            print("SELF COLLISION!!!")
            return True
        
        return False
        
    def update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def get_action(self, action):
        clock_wise_directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        index = clock_wise_directions.index(self.direction)

        # No direction change
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise_directions[index] # No Change
        # Right Turn
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_direction = clock_wise_directions[next_index] # Direction based on right turn
        # Left Turn
        else: # [0, 0, 1]
            next_index = (index - 1) % 4
            new_direction = clock_wise_directions[next_index] # Direction based on left turn
        return new_direction
        
    def _move(self, action):
        # [straight, right, left]

        self.direction = self.get_action(action)

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            
# User Input
# if __name__ == '__main__':
#     game = SnakeGame()
    
#     # game loop
#     while True:
#         game_over, score = game.play_step()
        
#         if game_over == True:
#             break
        
#     print('Final Score', score)
        
        
#     pygame.quit()