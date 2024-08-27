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

env_name = "reacher"
obs_dims, action_dims, _ = ENV_SPACES[env_name]

mlp_width = 32
mlp_depth=2
policy = rx.tasks.rl_policy.MLPPolicy(obs_dims, action_dims, mlp_width, mlp_depth, key=jr.key(1))
params, statics = eqx.partition(policy, eqx.is_array)
mdl_factory = lambda prms: eqx.combine(prms, statics)

task = rx.BraxTask(env_name, 500, mdl_factory)

logger = rx.Logger(wandb_log=True, metrics_fn=rx.logging.default_es_metrics, 
	ckpt_file="../ckpts_ex/es", ckpt_freq=50)

config = {
	"mlp_width": mlp_width,
	"mlp_depth": mlp_depth,
	"env": env_name
}

logger.init(project="es_brax_example", config=config) #useful only if logging to wandb
evolved_policy, state = rx.evolve(										#type:ignore
	params, 
	task, 
	key=jr.key(2), 
	strategy="DES",
	popsize=32,
	eval_reps=3, #number of evaluations per individual
	steps=256, 
	use_scan=False, # avoid using scan for memory
	fitness_shaper=ex.FitnessShaper(maximize=True),  #tell algorithm we want to maximize fitness (default is minimizing)
	logger=logger
)
logger.finish()