from casadi import MX

class Symbolics:
    def __init__(self, name):
        self.name = name
        self.q = None
        self.tau = None
        self.emg = None
        self.x = None
        self.p = None

    def get_symbol(self, name):
        return self.__dict__[name]

    def add_symbolics(self, name, size):
        self.__dict__[name] = MX.sym(name, size)