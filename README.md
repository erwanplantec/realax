# Realax
Set of wrappers and utilities for RL / evolutionary optimization / gradient optimization in JAX.

# Install

to install just clone the repo and then:

```
cd realax
python -m build
pip install dist/realax-0.0.2-py3-none-any.whl
```

Requirments are not handled by the package so far. It has been tested with JAX 0.4.28 (when installing jax , make sure that the PJRT plugin is not installed or remove it). 
Realax depends on the following libraries (you can install last versions for all, seem to work):
1. evosax
2. optax
3. gymnax
4. brax
5. tqdm
6. wandb


# Training

realax implements few wrappers allowing to optimize your models w.r.t to some tasks (i.e fitness/loss function) in a few lines and manage logging data to wandb while taking advantage of JAX acceleration.  

## ES

realax implements wrappers around evosax managing trinaing loops and data collection in an optimized way (scanned loops...). realax also supports multi device parrellization, you jjst have to pass the number of device the trainer should use and that's it (user should make sure the population size is actially divisible by the number of devices).

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

As with ES, any optimizer supported by optax can be passed to the trainer as a string but it can also manage any optimizer following the optax API by passing an instance of it instead of a string.

and as with ES we have a shorter function allowing to do the same training
```python
params, final_state, data = rx.optimize(
	initializer(jr.key(1)),
	rastrigin,
	jr.key(2),
	optimizer="adamw"
)
```

# Tasks

realax provides some wrappers for rl environments for them to be used with realax trainers. So far we have wrappers for gymnax environments with `GymnaxTask` and brax envs with `BraxTask`.
An example optimizing an mlp in a brax environment is given in the logging section. As for trainers, envs to be used can be specified with a string corresponding to one of brax/gymnax envs. However, an instance of an environment can also be passed so one can easily use custom environments following one of brax or gymnax apis.
Policies to be optimized on these tasks should inherit from `realax.tasks.rl_policy.BasePolicy` (or at least implement the `__call__` and `initialize` methods with the same signatures).

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

task = rx.BraxTask(
	env_name, 
	500, # max number of env steps
	mdl_factory, #function taking as input parameters and returning the model
	data_fn=lambda x:x, 
)

logger = rx.Logger(
	wandb_log=True, # if data should be logged to wandb
	metrics_fn=rx.training.log.default_es_metrics, # will log min, max and mean firness and ckpt current es mean
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

# TODO:
- utilities for easily using non-jittable functions (uncompiled training/env loops, callback utilities)
- Jumanji wrapper
- Fix the data collection issues for brax tasks


