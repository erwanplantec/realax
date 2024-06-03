from typing import Tuple, TypeAlias
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

PolicyState: TypeAlias = PyTree
EnvState: TypeAlias = PyTree
Observation: TypeAlias = jax.Array
RandomKey: TypeAlias = jax.Array
Action: TypeAlias = jax.Array

class BasePolicy(eqx.Module):

	#-------------------------------------------------------------------

	def __call__(self, obs: Observation, state: PolicyState, key: RandomKey)->Tuple[Action,PolicyState]:

		raise NotImplementedError

	#-------------------------------------------------------------------

	def initialize(self, key: RandomKey)->PolicyState:

		raise NotImplementedError

	#-------------------------------------------------------------------


class MLPPolicy(BasePolicy):
	#-------------------------------------------------------------------
	mlp: nn.MLP
	discrete_action: bool
	#-------------------------------------------------------------------
	def __init__(self, action_dims: int, obs_dims: int, width: int, depth: int,
		key: RandomKey, discrete_action: bool=True, **mlp_kws):

		self.mlp = nn.MLP(obs_dims, action_dims, width, depth, key=key, **mlp_kws)
		self.discrete_action = discrete_action

	#-------------------------------------------------------------------
	
	def __call__(self, obs: Observation, state: PolicyState, key: RandomKey) -> Tuple[Action, PolicyState]:
		
		a = self.mlp(obs)
		if self.discrete_action:
			a = jnp.argmax(a)
		return a, None
		
	#-------------------------------------------------------------------
	
	def initialize(self, key: RandomKey) -> PolicyState:
		return None
