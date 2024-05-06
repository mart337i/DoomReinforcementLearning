from classes.viz_doom_gym import VizDoomGym
from classes.train_and_logging_callback import TrainAndLoggingCallback
from classes.evaluation_of_trained_data import EvaluvatePPO
from classes.utils import Util
from classes.user_interface import UserInterface as Ui

from stable_baselines3 import PPO
from pick import pick
import os

CHECKPOINT_DIR = './train/train_corridor'
LOG_DIR = './logs/log_corridor'
CONFIG = '/home/sysadmin/code/doomReinforcementLearning/custom_senerios/deadly_corridor_custom_s1.cfg'
MODEL_DIR = "/home/sysadmin/code/doomReinforcementLearning/train/train_corridor"
util = Util()
ui = Ui()

def choose_option():
    title = 'Please Select an action: '
    options = ['Evaluate Env', 'Train model', 'Test model','Continue to learn on a model']
    option, index = pick(options, title)

    match index:
        case 0:
            ui.msg(f"You selected to {option}")
            env = VizDoomGym(render=True, config=CONFIG)
            res = util.check_env(env)
            if not res:
                ui.msg_confirm("Env was validated")
            else:
                ui.msg_confirm(res)

        case 1:
            ui.msg(f"You selected to {option}")
            title = 'Please select total_timesteps'
            options = {'1000': 1000, '10.000': 10000, '100.000': 100000, '500.000': 500000, '1.000.000': 1000000,'10.000.000' : 10000000 }
            option, index = pick(list(options.keys()), title)
            ui.timed_msg(f"The proces is starting now with {options.get(option)} total_timesteps")

            env = VizDoomGym(render=True, config=CONFIG)

            # # Agent & cretic
            model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00005, n_steps=2048)


            model.learn(total_timesteps=int(options.get(option)), callback=TrainAndLoggingCallback(check_freq=(round(abs(int(options.get(option))/10))), save_path=CHECKPOINT_DIR))

        case 2:
            title = "Select a Model to test:"
            options = []
            
            for path, currentDirectory, files in os.walk(MODEL_DIR):
                for file in files:
                    if not file.startswith("."):
                        options.append({
                            "name" : file.removesuffix('.zip'),
                            "path" : path + '/'
                        })
  
            if not options:
                return
            
            

            option, index = pick([item.get('name') for item in options], title)

            ui.timed_msg(f"You selected to {option}")
            env = VizDoomGym(render=True, config=CONFIG)
            print(options[index].get('path'))
            EvaluvatePPO(doomEnv=env,model=options[index].get('path') + option, number_of_episodes=20)

        case 3:
            title = "Select a Model to continue learning on:"
            options_paths = []
            
            for path, currentDirectory, files in os.walk(MODEL_DIR):
                for file in files:
                    if not file.startswith("."):
                        options_paths.append({
                            "name" : file.removesuffix('.zip'),
                            "path" : path + '/'
                        })
  
            if not options_paths:
                return
            
            option_path, path_index = pick([item.get('name') for item in options_paths], title)

            ui.timed_msg(f"You selected to {option_path}")

            title = 'Please select total_timesteps'
            amount_options = {'1000': 1000, '10.000': 10000, '100.000': 100000, '500.000': 500000, '1.000.000': 1000000}
            amount_option, index = pick(list(amount_options.keys()), title)
            ui.timed_msg(f"The proces is starting now with {amount_options.get(amount_option)} total_timesteps")

            env = VizDoomGym(render=True, config=CONFIG)
            # # Agent & cretic
            model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
            ppo_path_load = options_paths[path_index].get('path') + option_path
            model = model.load(ppo_path_load, env)
            model.learn(total_timesteps=int(amount_options.get(amount_option)), callback=TrainAndLoggingCallback(check_freq=(round(abs(int(amount_options.get(amount_option))/10))), save_path=CHECKPOINT_DIR))

        case _:
            ui.msg_confirm("Invalid option selected.")



def main():
    try:
        while True:
            choose_option()
    except KeyboardInterrupt:
        pass


main()