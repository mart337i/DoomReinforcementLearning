#!/usr/bin/env python

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import time

class EvaluvatePPO():

    def __init__(self,doomEnv):
        self.model = PPO.load('/home/sysadmin/code/doomReinforcementLearning/train/train_corridor/best_model_100000.zip')
        self.env = doomEnv
        self.mean_reward, _ = evaluate_policy(self.model, self.env, n_eval_episodes=10)

        self.evaluate()


    def evaluate(self):
       for episode in range(20): 
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done: 
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                time.sleep(0.02)
                total_reward += reward
            print('Total Reward for episode {} is {}'.format(total_reward, episode))
            time.sleep(2)