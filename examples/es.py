import realax as rx

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import wandb

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

logger = rx.Logger(True, rx.logging.default_es_metrics)

# 3. Run es
wandb.init(project="example", config=dict())
evolved_prms, _, data = rx.evolve(prms, rastrigin, jr.key(1), steps=32, logger=logger) #type:ignore
wandb.finish()

# 3. Plot data
fitnesses = data["metrics"]["fitness"]
plt.plot(fitnesses.mean(-1))
plt.title(f"$x^*=${np.array(evolved_prms)}")
plt.show()


