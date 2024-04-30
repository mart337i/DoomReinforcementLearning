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
util = Util()
ui = Ui()

def choose_option():
    title = 'Please Select an action: '
    options = ['Evaluate Env', 'Train model', 'Test model']
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
            options = {'1000': 1000, '10.000': 10000, '100.000': 100000, '500.000': 500000, '1.000.000': 1000000}
            option, index = pick(list(options.keys()), title)
            ui.msg(f"The proces is starting now with {option} total_timesteps")

            env = VizDoomGym(render=True, config=CONFIG)

            # # Agent & cretic
            model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192)


            model.learn(total_timesteps=int(option), callback=TrainAndLoggingCallback(check_freq=(int(option)/10), save_path=CHECKPOINT_DIR))

        case 2:
            ui.msg_confirm(f"You selected to {option}")
            env = VizDoomGym(render=True, config=CONFIG)
            EvaluvatePPO(env)

        case _:
            ui.msg_confirm("Invalid option selected.")



def main():
    try:
        while True:
            choose_option()
    except KeyboardInterrupt:
        pass


main()