try:
	from .brax_wrapper import BraxTask
except:
	pass
try:
	from .gymnax_wrapper import GymnaxTask
except:
	pass
from .rl_policy import BasePolicy

ENV_SPACES = {
	"CartPole-v1": (4, 2, "discrete"),
	"Acrobot-v1": (6, 3, "discrete"),
	"MountainCar-v0": (2, 3, "discrete"),
	"halfcheetah": (17, 6, "continuous"),
	"ant": (27, 8, "continuous"),
	"walker2d": (17, 6, "continuous"),
	"inverted_pendulum": (4, 1, "continuous"),
	'inverted_double_pendulum': (8, 1, "continuous"),
	"hopper": (11, 3, "continuous"),
	"Pendulum-v1": (3, 1, "continuous"),
	"PointRobot-misc": (6, 2, "continuous"),
	"MetaMaze-misc": (15, 4, "discrete"),
	"Reacher-misc": (8, 2, "continuous"),
	"reacher": (11, 2, "continuous")
}