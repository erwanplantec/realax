from .training import (
	evolve, 
	optimize, 
	EvosaxTrainer, 
	OptaxTrainer)
try:
	from .training import QDTrainer
except:
	pass
from .logging import Logger

try: from .tasks import GymnaxTask
except: pass

try: from .tasks import BraxTask
except: pass