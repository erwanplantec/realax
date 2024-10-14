from warnings import warn
from .es import EvosaxTrainer, evolve
from .grad import OptaxTrainer, optimize
try:
	from .qd import QDTrainer
except:
	warn("qd module cannot be imported")