try:
	from ._brax import BraxTask
except:
	pass
try:
	from ._gymnax import GymnaxTask
except:
	pass

def make_env(env: str, *args, **kwargs):
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
	for fctry in [BraxTask, GymnaxTask]: #type:ignore
		try:
			task = fctry(env, *args, **kwargs) 
			return task
		except:
			continue
	raise ValueError(f"environment {env} could not be created")


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
	"reacher": (11, 2, False)
}

def get_env_dimensions(env: str):
	if env.startswith("Synthetic-"):
		env = env.replace("Synthetic-", "")
	dims = ENV_SPACES.get(env, None)
	if dims is None:
		env = make_env(env).env
		#TODO
	return dims