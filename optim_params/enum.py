import enum

class Parameters(enum.Enum):
    """
    Enum class for the parameters to optimize.
    """
    f_iso = "f_iso"
    lm_optim = "lm_optim"
    lt_slack = "lt_slack"
