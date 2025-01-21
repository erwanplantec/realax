import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import evosax as ex
from typing import Optional, TypeAlias, Union
import chex
from flax import struct
from jaxtyping import PyTree

EvoParams: TypeAlias=PyTree

@struct.dataclass
class EvoState:
	mean: chex.Array
	archive: chex.Array
	fitness: chex.Array
	sigma: chex.Array
	best_member: chex.Array
	best_fitness: float = jnp.finfo(jnp.float32).max
	gen_counter: int = 0

def gaussian_mutation(x, sigma, key):
	return x + jr.normal(key, x.shape)*sigma

def point_mutations(x, sigma,  key):
	pass

def elementwise_crossover(a, b, crossover_rate, key):
	pass

def single_point_crossover(a, b, key):
	idx = jr.randint(key, (), 0, a.shape[0])
	msk = jnp.arange(a.shape[0]) <= idx
	return a * msk + b * (1-msk)

def tree_leaves_crossover(tree_a, tree_b, crossover_rate, key):
	pass


class GeneticAlgorithm(ex.Strategy):
	#-------------------------------------------------------------------
	def __init__(self, popsize: int, 
		num_dims: Optional[int] = None, pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None, 
		crossover_rate: float=0., elite_ratio: float=0.2, mutation_operator: str="gaussian", crossover_operator: str="elementwise",
		sigma_init: float=0.1, sigma_decay: float=1., sigma_limit: float=0.001,
		n_devices: Optional[int] = None, **fitness_kwargs: Union[bool, int, float]):

		self.strategy_name = "GA"

		super().__init__(
			popsize,
			num_dims,
			pholder_params,
			n_devices=n_devices,
			**fitness_kwargs
		)

		self.crossover_rate = crossover_rate
		self.elite_ratio = elite_ratio
		self.mutation_operator = mutation_operator
		self.crossover_operator = crossover_operator
		self.sigma_init = sigma_init
		self.sigma_decay = sigma_decay
		self.sigma_limit = sigma_limit
	#-------------------------------------------------------------------

	@property
	def params_strategy(self)->EvoParams:
		return None

	#-------------------------------------------------------------------
	
	def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
		return super().initialize_strategy(rng, params)

	#-------------------------------------------------------------------

	def ask_strategy(self, rng: chex.PRNGKey, state: EvoState, params: EvoParams) -> Tuple[chex.Array, EvoState]:
		return super().ask_strategy(rng, state, params)

	#-------------------------------------------------------------------

	def tell_strategy(self, x: chex.Array, fitness: chex.Array, state: EvoState, params: EvoParams) -> EvoState:
		return super().tell_strategy(x, fitness, state, params)