try:
	from ._brax import BraxTask
except:
	pass
try:
	from ._gymnax import GymnaxTask
except:
	pass
try:
	from ._kheperax import KheperaxTask
except :
	print("kheperax could not be loaded")
	pass
from .utils import MultiTaskAggregator

def make(env: str, *args, **kwargs):
	"""Make a task given by env name
	
	Args:
	    env (str): name of the environment to be created
	    *args: Description
	    **kwargs: Description
	
	Returns:
	    TYPE: Description
	
	Raises:
	    ValueError: Description
	"""
	try :
		tsk = BraxTask(env, *args, **kwargs) #type:ignore
	except:
		pass

	try:
		tsk = GymnaxTask(env, *args, **kwargs) #type:ignore
	except:
		pass

	try: 
		tsk = KheperaxTask(env, *args, **kwargs) #type:ignore
	except:
		raise ValueError(f"environment : {env} could not be created")
	return tsk


ENV_SPACES = {
	"CartPole-v1": (4, 2, True),
	"Acrobot-v1": (6, 3, True),
	"MountainCar-v0": (2, 3, True),
	"MountainCarContinuous-v0": (2, 1, False),
	"halfcheetah": (17, 6, False),
	"ant": (27, 8, False),
	"walker2d": (17, 6, False),
	"inverted_pendulum": (4, 1, False),
	'inverted_double_pendulum': (8, 1, False),
	"hopper": (11, 3, False),
	"Pendulum-v1": (3, 1, False),
	"PointRobot-misc": (6, 2, False),
	"MetaMaze-misc": (15, 4, True),
	"Reacher-misc": (8, 2, False),
	"reacher": (11, 2, False),
	"FourRooms-misc": (4, 4, True)
}

def get_env_dimensions(env: str):
	if env.startswith("Synthetic-"):
		env = env.replace("Synthetic-", "")
	dims = ENV_SPACES.get(env, None)
	if dims is None:
		env = make(env).env
		#TODO
		raise NotImplementedError(f"no dimensions given for {env}")
	return dims