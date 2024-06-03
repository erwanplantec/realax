# realax
Set of wrappers and utilities for JAX in the context of ML/ES/RL.

# Training

realax implements few wrappersallowing to optimize your models w.r.t to some tasks (i.e fitness/loss function) in a few lines.

## ES

```python
import realax as rx
import evosax as ex
import jax.random as jr
import jax.numpy as jnp

# 1. define your fitness function
def rastrigin(x, key=None, data=None):
	x = jnp.clip(x, -5.12, 5.12)
	n = x.shape[0]
	A = 10.
	y = A*n + jnp.sum(jnp.square(x) - A*jnp.cos(2*jnp.pi*x))
	return y, dict()

# 2. Set your model parameters structure
dims = 2
prms = jnp.zeros((dims,))

# 3. Instantiate your trainer
trainer = rx.Evosaxtrainer(
	train_steps=32, # number of training steps
	strategy="CMA_ES", #srategy to use: either a string corresponfing to one of evosax implemented strategy or a strategy following evosax API
	task=rastrigin,
	params_like=prms,  # structure of parameters (value is not used)	
	fitness_shaper=ex.FitnessShaper(maximize=False), #set to true if score should be maximized
	popsize=32,  # es population size
	eval_reps=1, # number of evaluation per indiviual (results are averages to get the individual fitness)
	n_devices=1. # number of devices to parallelize training over
)

# 3. Run es and colect data
final_es_state, data = trainer.init_and_train(jr.key(1))

# training loop can also be unrolled with fori_loop instead of scan allowing to save lot of memory (better when combined with wandb logging)
final_es_state = trainer.init_and_train_(jr.key(1)) # use fori_loop instaead of
```
EvosaxTriner can use any evosax strategy by either passing the name of the strategy or an actual instance of it as the `strategy` argument. It can also manage any strategy following the evosax API.

realax also provide a shorthand function to evolve parameters

```python
evolved_prms, final_es_state, data = evolve(
	prms, 
	rastrigin,
	jr.key(1),
	popsize=32,
	strategy="DES",
)
```

## Grad

API for gradient optimization is very similar to the ES one

```python
import realax as rx
import jax.numpy as jnp
import jax.random as jr

# 1. define your fitness function
def rastrigin(x, key=None, data=None):
	x = jnp.clip(x, -5.12, 5.12)
	n = x.shape[0]
	A = 10.
	y = A*n + jnp.sum(jnp.square(x) - A*jnp.cos(2*jnp.pi*x))
	return y, dict()

# 2. Set your model constructor
initializer = lambda key:  jr.normal(key, (2,))

# 3. Instamtiate trainer
trainer = rx.OptaxTrainer(
	epochs=64,
	optimizer="adamw",
	initializer=initializer,
	loss_fn=rastrigin,
    learning_rate=1e-3
)

# 4. Optimize params
final_state, data = trainer.init_and_train(jr.key(2)) #type:ignore

losses = data["metrics"]["loss"]
```

As for ES, anyb optimizer supported by optax can be passed to the trainer as a string but it can also manage any optimizer following the optax API by passing an instance of it instead of a string.

and as with ES we have a shorter function allowing to do the same trianing
```python
params, final_state, data = rx.optimize(
	initializer(jr.key(1)),
	rastrigin,
	jr.key(2),
	optimizer="adamw"
)
```

# Tasks

realax provides some wrappers for rl environemnts for them to be used with realax trainers. SO far we have wrappers for gymnax evnsironments with `GymnaxTask` and brax envs with `BraxTask`

# Logging

realax provides logging utilities allowing to save checkpoints during trainng and logging data to wandb. 

```python
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

logger = rx.Logger(
	wandb_log=True, 
	metrics_fn=rx.training.log.default_es_metrics, # will log min, max and mean firness and ckpt current es mean. 
	ckpt_file="../ckpts_ex/es", 
	ckpt_freq=50)

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
```


