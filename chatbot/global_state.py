class Borg:
    __shared_state = {}
    def __init__(self):
        self.__dict__ = self.__shared_state

class GlobalState(Borg):
    def __init__(self):
        Borg.__init__(self)