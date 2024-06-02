from realax.training.grad import optimize

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

# 1. define your fitness function
def rastrigin(x, key=None, data=None):
	x = jnp.clip(x, -5.12, 5.12)
	n = x.shape[0]
	A = 10.
	y = A*n + jnp.sum(jnp.square(x) - A*jnp.cos(2*jnp.pi*x))
	return y, dict()

# 2. Set your model parameters
prms = jr.normal(jr.key(1), (2,))

# 3. Optimize params
prms, _, data = optimize(prms, rastrigin, jr.key(1), steps=128) #type:ignore

# 4. Plot results
plt.plot(data["metrics"]["loss"])
plt.title(f"$x^*=${np.array(prms)}")
plt.show()
