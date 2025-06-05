import torch
import random
import numpy as np
from collections import deque
from snakeGameAI import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 10000
ALPHA = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0 # Number of games played
        self.epsilon = 0 # Exploration vs Exploitation measure
        self.gamma = 0.95 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # Exceeding memory removes elements in the left
        # model
        model_used = ""
        while model_used != "Y" and model_used != "N":
            model_used = input("Use previously saved model (Y/N)? ")
            
        self.model = Linear_QNet(11, 194, 3)
        if model_used == "Y":
            # SAVED THROUGH REWARD = 100 WEIGHTS
            self.model.load_state_dict(torch.load('./model/model.pth'))
            self.model.eval()
        # trainer
        self.trainer = QTrainer(self.model, alpha=ALPHA, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        neck = game.snake[1]

        # 4 points around the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Check which current game direction is
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food is left
            game.food.x > game.head.x,  # food is right
            game.food.y < game.head.y,  # food is up
            game.food.y > game.head.y,  # food is down
            
            # Distance to Food
            # game.distanceToFood(head.x, head.y) < game.distanceToFood(neck.x, neck.y)
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # pop left if MAX_MEMORY is reached/exceeded


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # Setting Exploration vs. Exploitation measure: Random actions to explore or capitalize the most optimal known action
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 225) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_record = 0
    agent = Agent()
    game = SnakeGame()
    reward = 0
    while True:
        # Retrieve current state
        old_state = agent.get_state(game)
        
        # Retrieve action
        final_action = agent.get_action(old_state)

        # Perform action and get new state
        reward, game_over, score = game.play_step(final_action, reward)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, final_action, reward, new_state, game_over)

        # remember
        agent.remember(old_state, final_action, reward, new_state, game_over)
        

        if game_over:
            # Train long memory & plot result
            print(reward)
            game.reset()
            
            agent.n_games += 1
            agent.train_long_memory()
            if score > best_record:
                best_record = score

                agent.model.save()

            reward = 0
            # Plotting
            plot_scores.append(score)
            total_score += score
            print('Game', agent.n_games, 'Score', score, 'Total_Score', total_score, 'Record', best_record)
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
