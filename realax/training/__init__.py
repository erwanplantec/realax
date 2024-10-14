from warnings import warn
from .evo import *
from .rl import *
from .grad import OptaxTrainer, optimize
try:
	from .qd import QDTrainer
except:
	warn("qd module cannot be imported")