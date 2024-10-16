import enum

class Parameters(enum.Enum):
    """
    Enum class for the parameters to optimize.
    """
    f_iso = "f_iso"
    l_optim = "l_optim"
    lt_slack = "lt_slack"
