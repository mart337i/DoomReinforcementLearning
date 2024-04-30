from stable_baselines3.common import env_checker

class Util():

    def check_env(self,env):
        # NOTE Validate ENV configuration 
        return env_checker.check_env(env=env)
    
