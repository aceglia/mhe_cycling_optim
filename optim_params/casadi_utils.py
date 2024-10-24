from casadi import MX, SX


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

    def add(self, name, size, sx=False):
        if sx:
            self.__dict__[name] = SX.sym(name, size)
        else:
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
        return_list = [self._get_one(n) for n in name]
        if len(return_list) == 1:
            return return_list[0]
        else:
            return return_list

    def _get_one(self, name):
        return self.__dict__[name]

    def add(self, name, data, sx=False):
        if sx:
            self.__dict__[name] = SX(data)
        else:
            self.__dict__[name] = MX(data)

    @staticmethod
    def to_mx(data):
        return MX(data)



