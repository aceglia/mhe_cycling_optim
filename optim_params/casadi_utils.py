from casadi import MX


class Symbolics:
    def __init__(self):
        self.q = None
        self.tau = None
        self.emg = None
        self.x = None
        self.p = None

    def get(self, name):
        name = [name] if not isinstance(name, list) else name
        return [self._get_one(n) for n in name]

    def _get_one(self, name):
        return self.__dict__[name]

    def add_symbolics(self, name, size):
        self.__dict__[name] = MX.sym(name, size)


class MxVariables:
    def __init__(self):
        self.q = None
        self.dq = None
        self.ddq = None
        self.tau = None
        self.emg = None
        self.x = None
        self.p = None
        self.f_ext = None

    def get(self, name):
        name = [name] if not isinstance(name, list) else name
        return [self._get_one(n) for n in name]

    def _get_one(self, name):
        return self.__dict__[name]

    def add_variable(self, name, data):
        self.__dict__[name] = MX(data)

    @staticmethod
    def to_mx(data):
        return MX(data)



