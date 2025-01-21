"""Summary
"""
from typing import Callable, NamedTuple, Optional, Tuple, TypeAlias, Union
import jax
import jax.random as jr
import equinox as eqx

from brax import envs
from brax.envs import Env
from jaxtyping import Float, PyTree

Params: TypeAlias = PyTree
TaskParams: TypeAlias = PyTree
EnvState: TypeAlias = PyTree
Action: TypeAlias = jax.Array
PolicyState: TypeAlias = PyTree
BraxEnv: TypeAlias = Env


class State(NamedTuple):
	"""Summary
	"""
	env_state: EnvState
	policy_state: PolicyState


class BraxTask(eqx.Module):
	#-------------------------------------------------------------------
	env: BraxEnv
	model_factory: Callable
	max_steps: int
	data_fn: Callable[[PyTree], dict]
	#-------------------------------------------------------------------
	def __init__(
		self, 
		env: Union[str, BraxEnv],
		max_steps: int=100,
		model_factory: Callable=lambda params: params,
		backend: str="positional",
		data_fn: Callable=lambda x: x, 
		env_kwargs: dict={}):
		"""Summary
		
		Args:
		    env (Union[str, BraxEnv]): Description
		    max_steps (int): Description
		    model_factory (Callable, optional): Description
		    backend (str, optional): Description
		    data_fn (Callable, optional): Description
		    env_kwargs (dict, optional): Description
		"""
		if isinstance(env, str):
			self.env = envs.get_environment(env, backend=backend, **env_kwargs)
		else:
			self.env = env

		self.model_factory = model_factory
		self.max_steps = max_steps
		self.data_fn = data_fn

	#-------------------------------------------------------------------

	def __call__(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[Float, PyTree]:
		"""Summary
		
		Args:
		    params (Params): Description
		    key (jax.Array): Description
		    task_params (Optional[TaskParams], optional): Description
		
		Returns:
		    Tuple[Float, PyTree]: Description
		"""
		_, ret = self.rollout(params, key, task_params)
		return ret, {}

	#-------------------------------------------------------------------

	def rollout(
		self, 
		params: Params, 
		key: jax.Array, 
		task_params: Optional[TaskParams]=None)->Tuple[State, Float]:
		"""Summary
		
		Args:
		    params (Params): Description
		    key (jax.Array): Description
		    task_params (Optional[TaskParams], optional): Description
		
		Returns:
		    Tuple[State, Float]: Description
		"""
		key, init_env_key, init_policy_key, rollout_key = jr.split(key, 4)
		policy = self.model_factory(params)
		
		policy_state = policy.initialize(init_policy_key)
		env_state = self.initialize(init_env_key)
		init_state = State(env_state=env_state, policy_state=policy_state)

		def env_step(carry):
			"""Summary
			
			Args:
			    carry (TYPE): Description
			
			Returns:
			    TYPE: Description
			"""
			state, ret, steps, key = carry
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state, _key)
			env_state = self.env.step(state.env_state, action)
			ret = ret + env_state.reward
			new_state = State(env_state=env_state, policy_state=policy_state)
			return [new_state, ret, steps+1, key]

		#[state, _] = jax.lax.scan(env_step, [init_state, 1., rollout_key], None, self.max_steps)
		state, ret, *_ = jax.lax.while_loop(
			lambda c: (c[0].env_state.done==0.)&(c[2]<self.max_steps), 
			env_step, 
			[init_state, 0., 0, rollout_key], 
		)
		
		return state, ret

	#-------------------------------------------------------------------

	def step(self, *args, **kwargs):
		"""Summary
		
		Args:
		    *args: Description
		    **kwargs: Description
		
		Returns:
		    TYPE: Description
		"""
		return self.env.step(*args, **kwargs)

	def reset(self, *args, **kwargs):
		"""Summary
		
		Args:
		    *args: Description
		    **kwargs: Description
		
		Returns:
		    TYPE: Description
		"""
		return self.env.reset(*args, **kwargs)

	#-------------------------------------------------------------------

	def initialize(self, key:jax.Array)->EnvState:
		"""Summary
		
		Args:
		    key (jax.Array): Description
		
		Returns:
		    EnvState: Description
		"""
		return self.env.reset(key)

	#-------------------------------------------------------------------