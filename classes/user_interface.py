import time


class UserInterface():
    
    def msg_confirm(self,msg: str):
        print(msg)
        input()

    def msg(self,msg: str):
        print(msg)

    def timed_msg(self,msg: str,seconds: int = 2):
        print(msg)
        time.sleep(2)