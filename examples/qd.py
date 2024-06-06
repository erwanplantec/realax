import jax.numpy as jnp
import jax.random as jr
import realax as rx
import matplotlib.pyplot as plt

from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.utils.plotting import plot_2d_map_elites_repertoire

def rastrigin(x, key=None, data=None):
	x = jnp.clip(x, -5.12, 5.12)
	n = x.shape[0]
	A = 10.
	y = A*n + jnp.sum(jnp.square(x) - A*jnp.cos(2*jnp.pi*x))
	return -y, x/5.12, dict()


centroids = compute_euclidean_centroids((16,16), minval=-1., maxval=1.)
emitter = CMAImprovementEmitter(64, 2, centroids, 0.1)
emitter = CMAPoolEmitter(16, emitter)

qd = rx.training.QDTrainer(emitter, rastrigin, 256, jnp.zeros((2,)), centroids=centroids)

state = qd.init_and_train_(jr.key(1))
rep = state.repertoire
fit = rep.fitnesses

fig, ax = plt.subplots()
plot_2d_map_elites_repertoire(centroids, fit, -1., 1., rep.descriptors, ax) #type:ignore
plt.show()