import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.random as jr
from jaxtyping import PyTree
import evosax as ex
import matplotlib.pyplot as plt

import realax as rx
from realax.tasks import ENV_SPACES

class MLPPolicy(eqx.Module):
	#-------------------------------------------------------------------
	# Parameters:
	mlp: nn.MLP
	#-------------------------------------------------------------------

	def __init__(self, obs_dims: int, action_dims: int, key: jax.Array):
		self.mlp = nn.MLP(obs_dims, action_dims, 64, 1, 
			final_activation=jnn.tanh, key=key)

	#-------------------------------------------------------------------

	def __call__(self, obs: jax.Array, state: PyTree, key: jax.Array):
		
		a = self.mlp(obs)
		return a, None

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array):
		return None


env_name = "reacher"
obs_dims, action_dims, _ = ENV_SPACES[env_name]

policy = MLPPolicy(obs_dims, action_dims, key=jr.key(1))
params, statics = eqx.partition(policy, eqx.is_array)
mdl_factory = lambda prms: eqx.combine(prms, statics)

task = rx.BraxTask(env_name, 500, mdl_factory)

logger = rx.Logger(wandb_log=True, metrics_fn=rx.training.log.default_es_metrics, 
	ckpt_file="../ckpts_ex/es", ckpt_freq=50)

logger.init(project="es_brax_example", config={"env":"reacher"})
evolved_policy, state=evolve(										#type:ignore
	params, 
	task, 
	key=jr.key(2), 
	strategy="CMA_ES",
	popsize=32,
	eval_reps=3, #number of evaluations per individual
	steps=256, 
	use_scan=False, # avoid using scan for memory
	fitness_shaper=ex.FitnessShaper(maximize=True),  #tell algorithm we want to maximize fitness (default is minimizing)
	logger=logger
)
logger.finish()