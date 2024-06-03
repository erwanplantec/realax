import realax as rx

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

# 2. Set your model constructor
initializer = lambda key:  jr.normal(key, (2,))

# 3. Instamtiate trainer
trainer = rx.OptaxTrainer(
	epochs=64,
	optimizer="adamw",
	initializer=initializer,
	loss_fn=rastrigin
)

# 4. Optimize params
final_state, data = trainer.init_and_train(prms, rastrigin, jr.key(1), steps=128) #type:ignore

# 4. Plot results
plt.plot(data["metrics"]["loss"])
plt.title(f"$x^*=${np.array(final_state.params)}")
plt.show()
