#Imports
#Vizdoom for ZDoom
from vizdoom import *
#random for random decisions
import random
#time for sleeping between actions and games, maybe tracking time too
import time
#OpenAI gym environment  to wrap vizdoom in
from gym import Env
#Import gym spaces
from gym.spaces import Discrete, Box
#for grayscaling, maybe not needed?
import cv2
#data types
import numpy as np
#For graphing and showing image
from matplotlib import pyplot as plt
#for navigation
import os
#for saving the model as it trains
from stable_baselines3.common.callbacks import BaseCallback
#Import algorithm PPO
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3

#check policy and environment to ensure they are valid and functional
from stable_baselines3.common import env_checker
from stable_baselines3.common.evaluation import evaluate_policy

#DoomGameEnv class to instantiate games and track variables
#Wrapped in the OpenAI Gym environment
class DoomGameEnv(Env):

    #Do everything to start an environment
    def __init__(self, scenario, Algo, render = False, load = "", loadpath = ""):
        super().__init__()
        #Instantiate a DoomGame using the config file "basic.cfg"
        #"Basic.cfg" contains plenty of data about settings the game will run under
        #can be adjusted later
        self.Load = load
        self.AlgoClass = Algo
        #getting class of algo and string of algo, both are useful
        spaceSplit = str(self.AlgoClass).split(" ")
        dotSplit = spaceSplit[1].split(".")
        aposSplit = dotSplit[-1].split("'")
        self.AlgoString = aposSplit[0]
        #save directories for models and training files
        self.CheckpointDir = "./Train/Train/Curriculum" + str(scenario) + "/" + str(self.AlgoString)
        self.LogDir = "./Logs/Log/Curriculum" + str(scenario) + "/" + str(self.AlgoString)
        #level
        self.Scenario = scenario.lower()
        self.cfgPath = "ViZDoom-1.1.11/scenarios/" + str(self.Scenario) + ".cfg"
        #how to save the data. Param 1 dictates how many iterations before
        #taking the best model of that iteration
        self.Callback = TrainAndLoggingCallback(10000, self.CheckpointDir)
        #testing parameters, for when load is active
        self.Episodes = 20
        self.EpisodicSleep = 2
        self.IntraRunSleep = 0.2
        #instantiate environment
        self.game = DoomGame()
        self.game.load_config(self.cfgPath)
        #get num actions
        self.numActions = len(list(self.game.get_available_buttons()))

        #some params to handle curriculum learning and reward shaping, advised not to touch
        #unless base params of the config changes too.
        self.health = 100
        self.damage_taken = 0
        self.hit_count = 0
        self.ammo = 52

        #set up the action matrix of current possible actions
        self.actions = np.identity(self.numActions, dtype = np.uint8)

        #decide if the display should be shown to speed up learning
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        #start the game
        self.game.init()

        #If not resizing the image, this is unnecessary I think?
        #creates an observation_space, defining the size of the viewport and remaking a screen
        self.observation_space = Box(low = 0, high = 255, shape = (3, 240, 320), dtype = np.uint8)
        #self.observation_space = Box(low = 0, high = 255, shape = (100, 160, 1) dtype = np.uint8)
        self.action_space = Discrete(self.numActions)

        #if you want to see the model, its assumed you are testing it!
        if self.render == True:
            print(str(self.CheckpointDir) + "/" + "Best_Model_" + str(load))
            self.Model = self.AlgoClass.load(str(self.CheckpointDir) + "/" + "Best_Model_" + str(load))
            self.TestModel(self.Episodes, self.EpisodicSleep, self.IntraRunSleep)
        #try to load model if given a path. if no path, load with settings below.
        #these hyperparameters are fragile and small changes can have huge impact
        #see report to understand more about what these are.
        else:
            if loadpath == "":
                self.Model = self.AlgoClass(
                'CnnPolicy',
                self,
                tensorboard_log=self.LogDir,
                verbose=1,
                n_steps = 2048,
                learning_rate = 0.00005,
                gamma = 0.95,
                gae_lambda = 0.5,
                ent_coef = 0.001,
                vf_coef = 0.5,
                max_grad_norm = 0.5)
            else:
                self.Model = self.AlgoClass.load(loadpath, env = self)

            #specify how many iterations to run the env for and what save callback to use
            self.Model.learn(total_timesteps = 200000, callback = self.Callback)

            self.close()

    #make an action in the environment
    def step(self, action):
        #get the game reward for taking the action with frame skip parameter
        movementreward = self.game.make_action(self.actions[action], 4)


        reward = 0
        #get information about the state
        if self.game.get_state():
            image = self.game.get_state().screen_buffer
            #image = self.greyscale(image)
            #ammo = self.game.get_state().game_variables[0]

            gamevariables = self.game.get_state().game_variables
            health, damage_taken, hit_count, ammo = gamevariables

            #perform reward shaping
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hit_count_delta = hit_count - self.hit_count
            self.hit_count = hit_count
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            #can tweak the multipliers to make actions more or less meaningful to the agent.
            #these values seem to cause volatility if changed too much away from current settings
            reward = movementreward + damage_taken_delta*10 + hit_count_delta*200 + ammo_delta*5

            gameVars = ammo
        else:
            #game vars and image are expected, just putting something down.
            image = np.zeros(self.observation_space.shape)
            gameVars = 0

        gameVars = {"gameVars":gameVars}

        done = self.game.is_episode_finished()

        return image, reward, done, gameVars

    #to close the idle game when completed
    def close(self):
        self.game.close()

    #this function is predefined by vizdoom
    #but used to render the game
    def render():
        #this is handled by vizdoom
        pass

    #Greyscale the game and resize it
    '''def greyscale(self, observation):
        #move the colour channels to be last instead of first, what cv2 expects
        np.moveaxis(observation,0,-1)
        grey = cv2.cvtColor(observation, 0, -1), cv2.COLOR_BGR2GAY)
        resize = cv2.resize(grey, (160, 100), interpolation = cv2.INTER_CUBIC)
        image = np.reshape(resize, 100, 160, 1)
        return image'''

    #What happens when a new game begins. A game is not an environment
    def reset(self):
        #make a new episode under the current settings
        self.game.new_episode()
        image = self.game.get_state().screen_buffer
        #image = self.greyscale(image)

        return image


    #run a default test loop.
    def TestModel(self, episodes, EpisodicSleep, IntraRunSleep):

        mean_reward, _ = evaluate_policy(self.Model, self, n_eval_episodes = 100)

        for episode in range(episodes):
            obs = self.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.Model.predict(obs)
                obs, reward, done, info = self.step(action)
                time.sleep(IntraRunSleep)
                total_reward += reward
            print("Total reward: " + str(total_reward) + " episode " + str(episode))
            time.sleep(EpisodicSleep)

#A class to handle how to save model data and where to.
#Credit to Nicholas Renotte and his excellent youtube video where he explains this around 1 hour 17 minutes in
#https://youtu.be/eBCU-tqLGfQ?t=4633

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, CheckFrequency, SavePath, Verbose =1):
        super(TrainAndLoggingCallback, self).__init__(Verbose)
        self.CheckFrequency = CheckFrequency
        self.SavePath = SavePath

    def _init_callback(self):

        if self.SavePath is not None:
            os.makedirs(self.SavePath, exist_ok = True)

    def _on_step(self):
        if self.n_calls % self.CheckFrequency == 0:
            ModelPath = os.path.join(self.SavePath, "Best_Model_" + str(self.n_calls))
            self.model.save(ModelPath)

#A list of cfg files with camelcase + underscores to match cfg file names
#for ease of imports
Scenarios = [
'Basic',
'Bots',
'Cig_With_Unknown',
'Cig',
'Deadly_Corridor_1',
'Deadly_Corridor_2',
'Deadly_Corridor_3',
'Deadly_Corridor_4',
'Deadly_Corridor',
'Deathmatch',
'Defend_The_Center',
'Defend_The_Line',
'Health_Gathering_Supreme',
'Health_Gathering',
'Learning',
'Multi_Deathmatch',
'Multi_Duel',
'Multi',
'My_Way_Home',
'Oblige',
'Perfect_Bots',
'Predict_Position',
'Rocket_Basic',
'Simpler_Basic',
'Take_Cover'
]

#A list of possible algos for stable baselines 3 (unsure if they take same params)
Algos = [
A2C,
DQN,
HER,
PPO,
SAC,
TD3
]
#param 1 is the scenario, 2 is what algorithm (ppo, a2c, dqn are used for this mostly)
#param 4 is whether or not you are testing
#param 5 is what model to load for the current settings
#param 6 is a specific load path for settings outside current, useful for transfer and curriculum learning
env = DoomGameEnv(Scenarios[4], Algos[3], render = False, load = "", loadpath = "")

env = DoomGameEnv(Scenarios[5], Algos[3], render = False, load = "", loadpath = "./Train/Train/CurriculumDeadly_Corridor_1/PPO/Best_Model_200000.zip")

env = DoomGameEnv(Scenarios[6], Algos[3], render = False, load = "", loadpath = "./Train/Train/CurriculumDeadly_Corridor_2/PPO/Best_Model_200000.zip")

env = DoomGameEnv(Scenarios[7], Algos[3], render = False, load = "", loadpath = "./Train/Train/CurriculumDeadly_Corridor_3/PPO/Best_Model_200000.zip")

env = DoomGameEnv(Scenarios[8], Algos[3], render = False, load = "", loadpath = "./Train/Train/CurriculumDeadly_Corridor_4/PPO/Best_Model_200000.zip")

#env = DoomGameEnv(Scenarios[4], Algos[3], render = True, load = "10000", loadpath = "")

#checks the environment is valid.
env_checker.check_env(env)

'''#These arrays map to binary 1's and 0's of active states
#of the buttons being pressed.
#0,0,1 is only left mouse button in this case.
#could these be combined into 0,1,1 to move right AND shoot?
shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

#Loop through 10 episodes
episodes = 10
for i in range(episodes):
    #start new episode
    game.new_episode()
    #check if the game is over
    while not game.is_episode_finished():
        #get inforamtion about the game
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        #find reward of the action taken
        #This second parameter fixes frame skip. Skip 4 frames and see the reward
        #of the 4 actions taken.
        #what if you fire, but the projectile is slow? does it hit or not?
        reward = game.make_action(random.choice(actions), 4)
        print("reward:", reward)
        #give time for the game to process the action as its not instant
        #Must consider "frame skip"
        #actions take time to show. if it were instant, button release
        #is also instant.
        time.sleep(0.02)
    #reward of the whole game
    print("Result:", game.get_total_reward())
    #between game sleep time
    time.sleep(2)
'''