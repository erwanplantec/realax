# realax
Set of wrappers and utilities for JAX in the context of ML/ES/RL.

# Training

realax implements few wrappersallowing to optimize your models w.r.t to some tasks (i.e fitness/loss function) in a few lines.

## ES

'''python
import realax as rx

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

# 3. Run es
evolved_prms, _, data = rx.evolve(prms, rastrigin, jr.key(1), steps=32)
'''

## Grad


# Logging
