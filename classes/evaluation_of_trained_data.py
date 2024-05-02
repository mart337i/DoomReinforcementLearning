#!/usr/bin/env python

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from .user_interface import UserInterface

import time

class EvaluvatePPO():

    def __init__(self,doomEnv,model,number_of_episodes):
        self.model = PPO.load(model)
        self.env = doomEnv
        self.mean_reward, _ = evaluate_policy(self.model, self.env, n_eval_episodes=10)
        self.ui = UserInterface()
        self.number_of_episodes = number_of_episodes
        
        # Run eval
        self.evaluate()


    def evaluate(self):
        episode_scores = []
        for episode in range(self.number_of_episodes): 
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done: 
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                time.sleep(0.02)
                total_reward += reward
            print(f'Total Reward for episode {episode} is {total_reward}')
            episode_scores.append(total_reward)
            time.sleep(2)

        self.ui.msg_confirm(f'Avg reward for episodes : {abs(sum(episode_scores)) / len(episode_scores)}')