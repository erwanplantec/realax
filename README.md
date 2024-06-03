# realax
Set of wrappers and utilities for JAX in the context of ML/ES/RL.

# Training

realax implements few wrappersallowing to optimize your models w.r.t to some tasks (i.e fitness/loss function) in a few lines.

## ES

```python
import realax as rx
import evosax as ex
import jax.random as jr

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

# 3. Run es
final_es_state, data = trainer.init_and_train(jr.key(1))
```


## Grad


# Logging
