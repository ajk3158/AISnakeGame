import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from gymnasium import spaces
import sys, pygame
from helper import plot
from snakeGameAI import SnakeGame, Point, Direction
from stable_baselines3.common.env_checker import check_env
from collections import deque

REMEMBERED_ACTIONS = 5
class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.game = SnakeGame()
        self.snake = self.game.snake
        self.food = None
        self.currReward = 0
        self.score = 0
        self.snake_clock_wise_directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # All possible discrete actions completed by agent:
        # num_actions = 4 
        # num_observations = 4 # Snake x, Snake y, Food x, Food y
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        # What are the things we are observing in our environment that affects our agent
        obs_size = 4 + REMEMBERED_ACTIONS
        self.observation_space = spaces.Box(low=-500, high=500, shape=(obs_size, ), dtype=np.float32)

    def step(self, actionIndex):
        actiongrid = [0, 0, 0]
        if actionIndex != 3:
            actiongrid[actionIndex] = 1
        action = self.game.get_action(actiongrid)
        self.prev_actions.append(self.snake_clock_wise_directions.index(action))
        returnedTuple = self.game.play_step(actiongrid, self.currReward, sbUsed=True)
        # In returned tuple: snake, food, reward, terminated, truncated, score
        self.snake = returnedTuple[0]
        self.food = returnedTuple[1]
        self.currReward = returnedTuple[2]
        game_over = returnedTuple[3]
        time_up = returnedTuple[4]
        self.score = returnedTuple[5]
        info = {}
        observation = [self.snake[0].x, self.snake[0].y, self.food.x, self.food.y] + list(self.prev_actions)
        # for prev_action in self.prev_actions:
        #     observation.append(prev_action)
        observation = np.array(observation)
        return observation, self.currReward, game_over, time_up, info

    def reset(self, seed=None):
        self.game.reset()
        self.snake = self.game.snake
        self.food = self.game.food
        self.currReward = 0
        self.score = 0
        self.prev_actions = deque(maxlen=REMEMBERED_ACTIONS)
        for i in range(REMEMBERED_ACTIONS):
            self.prev_actions.append(-1)
        
        observation = [self.snake[0].x, self.snake[0].y, self.food.x, self.food.y] + list(self.prev_actions)
        # for prev_action in self.prev_actions:
        #     observation.append(prev_action)
        observation = np.array(observation)
        return observation, {}
    
    def render(self):
        pass

    def close(self):
        pygame.quit()
        sys.exit()
        

# Assigning environment to variable
env = SnakeEnv()

# Initialize model (which have policies built from the environment)
# model = PPO("MlpPolicy", env, verbose=1)
# Training model with the given environment
# model.learn(total_timesteps=10000)

# Proximal Policy Optimization (utilizes multiple workers from A2C and trust region to improve actor from TRPO)
model = PPO("MlpPolicy", env, verbose=1, n_epochs=15, ent_coef=0.001, learning_rate=0.0001, gamma=0.99, batch_size=256)
model.learn(total_timesteps=25000)
model.save(path="snake_PPO_model")

# TIMESTEPS = 10000
# iters = 0s
# while True:
# 	iters += 1
# 	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
# 	model.save(path="snake_PPO_model")
# Iterating for certain number of steps
# vec_env = model.get_env()
plot_scores = []
plot_mean_scores = []
total_score = 0
best_record = 0
n_games = 0
obs, info = env.reset()
while True:
    action, state = model.predict(obs)
    print(action)
    obs, reward, game_over, time_up, info = env.step(action)
    # env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
    # train short memory
    
    if game_over or time_up:
        # Train long memory & plot result
        print(env.score)
        n_games += 1
        if env.score > best_record:
            best_record = env.score
            model.save()
        # Plotting
        plot_scores.append(env.score)
        total_score += env.score
        print('Game', n_games, 'Score', env.score, 'Total_Score', total_score, 'Record', best_record)
        mean_score = total_score / n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
        obs, info = env.reset()