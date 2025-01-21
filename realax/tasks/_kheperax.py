from kheperax.tasks.config import KheperaxConfig
from kheperax.tasks.main import KheperaxTask as _KheperaxTask
from kheperax.tasks.target import TargetKheperaxTask, TargetKheperaxConfig
from kheperax.tasks.quad import make_quad_config
from kheperax.simu.robot import Robot
from kheperax.simu.maze import Maze
from kheperax.envs.env import Env
from kheperax.envs.maze_maps import register_target_maze

import jax.numpy as jnp
import jax.random as jr
import jax
import equinox as eqx
from jaxtyping import PyTree, Float
from typing import Callable, TypeAlias, Optional, NamedTuple, Tuple


register_target_maze(
	"empty", 
	segments = [],
	target_pos=(0.15, 0.9),
	target_radius=0.05)

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


class KheperaxTask(eqx.Module):
	#-------------------------------------------------------------------
	env: Env
	model_factory: Callable
	max_steps: int
	data_fn: Callable[[PyTree], dict]
	#-------------------------------------------------------------------
	def __init__(
		self, 
		maze: str,
		has_target: bool=False,
		max_steps: int=250,
		model_factory: Callable=lambda params: params,
		data_fn: Callable=lambda x: x, 
		config_kwargs: dict={},
		robot_kwargs: dict={}):
		"""Summary
		
		Args:
		    env (Union[str, BraxEnv]): Description
		    max_steps (int): Description
		    model_factory (Callable, optional): Description
		    backend (str, optional): Description
		    data_fn (Callable, optional): Description
		    env_kwargs (dict, optional): Description
		"""
		is_quad = "-quad" in maze
		maze = maze.replace("-quad", "")
		Config = TargetKheperaxConfig if has_target else KheperaxConfig
		Task = TargetKheperaxTask if has_target else _KheperaxTask
		config = Config.get_default_for_map(maze)
		robot = Robot.create_default_robot().replace(**robot_kwargs)
		config.robot = robot
		if is_quad:
			config = make_quad_config(config)
		env, *_ = Task.create_default_task(config, random_key=jr.key(1))
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
		ret, data = self.rollout(params, key, task_params)
		return ret, data

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

		def env_step(carry, x):
			"""Summary
			
			Args:
			    carry (TYPE): Description
			
			Returns:
			    TYPE: Description
			"""
			state, ret, steps, valid, key = carry
			key, _key = jr.split(key)
			action, policy_state = policy(state.env_state.obs, state.policy_state, _key)
			env_state = self.env.step(state.env_state, action)
			ret = ret + env_state.reward * valid
			new_state = State(env_state=env_state, policy_state=policy_state)
			valid = valid * (1-new_state.env_state.done)
			return [new_state, ret, steps+(1-state.env_state.done), valid, key], state

		[final_state, ret, _, _, _], states = jax.lax.scan(
			env_step, [init_state, 0., 0, 1., rollout_key], None, self.max_steps)

		final_pos = final_state.env_state.robot.posture
		bd = jnp.array([final_pos.x, final_pos.y])

		return ret, {"states":  states, "bd": bd, "final_state": final_state}

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


# class KheperaxQDTask(eqx.Module):
# 	#-------------------------------------------------------------------
# 	env: Env
# 	model_factory: Callable
# 	max_steps: int
# 	data_fn: Callable[[PyTree], dict]
# 	#-------------------------------------------------------------------
# 	def __init__(
# 		self, 
# 		maze: str,
# 		has_target: bool=False,
# 		max_steps: int=250,
# 		model_factory: Callable=lambda params: params,
# 		data_fn: Callable=lambda x: x, 
# 		config_kwargs: dict={},
# 		robot_kwargs: dict={}):
# 		"""Summary
		
# 		Args:
# 		    env (Union[str, BraxEnv]): Description
# 		    max_steps (int): Description
# 		    model_factory (Callable, optional): Description
# 		    backend (str, optional): Description
# 		    data_fn (Callable, optional): Description
# 		    env_kwargs (dict, optional): Description
# 		"""
# 		is_quad = "-quad" in maze
# 		maze = maze.replace("-quad", "")
# 		Config = TargetKheperaxConfig if has_target else KheperaxConfig

# 		config = Config.get_default_for_map(maze)
# 		robot = Robot.create_default_robot().replace(**robot_kwargs)
# 		config.robot = robot
# 		if is_quad:
# 			config = make_quad_config(config)
# 		env, *_ = _KheperaxTask.create_default_task(config, random_key=jr.key(1))
# 		self.env = env
# 		self.model_factory = model_factory
# 		self.max_steps = max_steps
# 		self.data_fn = data_fn

# 	#-------------------------------------------------------------------

# 	def __call__(
# 		self, 
# 		params: Params, 
# 		key: jax.Array, 
# 		task_params: Optional[TaskParams]=None)->Tuple[Float, Float, PyTree]:
# 		"""Summary
		
# 		Args:
# 		    params (Params): Description
# 		    key (jax.Array): Description
# 		    task_params (Optional[TaskParams], optional): Description
		
# 		Returns:
# 		    Tuple[Float, PyTree]: Description
# 		"""
# 		ret, data = self.rollout(params, key, task_params)
# 		bd = data["bd"]
# 		return ret, bd, data

# 	#-------------------------------------------------------------------

# 	def rollout(
# 		self, 
# 		params: Params, 
# 		key: jax.Array, 
# 		task_params: Optional[TaskParams]=None)->Tuple[State, Float]:
# 		"""Summary
		
# 		Args:
# 		    params (Params): Description
# 		    key (jax.Array): Description
# 		    task_params (Optional[TaskParams], optional): Description
		
# 		Returns:
# 		    Tuple[State, Float]: Description
# 		"""
# 		key, init_env_key, init_policy_key, rollout_key = jr.split(key, 4)
# 		policy = self.model_factory(params)
		
# 		policy_state = policy.initialize(init_policy_key)
# 		env_state = self.initialize(init_env_key)
# 		init_state = State(env_state=env_state, policy_state=policy_state)

# 		def env_step(carry, x):
# 			"""Summary
			
# 			Args:
# 			    carry (TYPE): Description
			
# 			Returns:
# 			    TYPE: Description
# 			"""
# 			state, ret, steps, valid, key = carry
# 			key, _key = jr.split(key)
# 			action, policy_state = policy(state.env_state.obs, state.policy_state, _key)
# 			env_state = self.env.step(state.env_state, action)
# 			ret = ret + env_state.reward * valid
# 			new_state = State(env_state=env_state, policy_state=policy_state)
# 			valid = valid * (1-new_state.env_state.done)
# 			return [new_state, ret, steps+(1-state.env_state.done), valid, key], state

# 		[final_state, ret, _, _, _], states = jax.lax.scan(
# 			env_step, [init_state, 0., 0, 1., rollout_key], None, self.max_steps)

# 		final_pos = final_state.env_state.robot.posture
# 		bd = jnp.array([final_pos.x, final_pos.y])

# 		return ret, {"states":  states, "bd": bd, "final_state": final_state}

# 	#-------------------------------------------------------------------

# 	def step(self, *args, **kwargs):
# 		"""Summary
		
# 		Args:
# 		    *args: Description
# 		    **kwargs: Description
		
# 		Returns:
# 		    TYPE: Description
# 		"""
# 		return self.env.step(*args, **kwargs)

# 	def reset(self, *args, **kwargs):
# 		"""Summary
		
# 		Args:
# 		    *args: Description
# 		    **kwargs: Description
		
# 		Returns:
# 		    TYPE: Description
# 		"""
# 		return self.env.reset(*args, **kwargs)

# 	#-------------------------------------------------------------------

# 	def initialize(self, key:jax.Array)->EnvState:
# 		"""Summary
		
# 		Args:
# 		    key (jax.Array): Description
		
# 		Returns:
# 		    EnvState: Description
# 		"""
# 		return self.env.reset(key)

# 	#-------------------------------------------------------------------


if __name__ == '__main__':
	import equinox.nn as nn
	import numpy as np

	class Policy(eqx.Module):
		mlp: nn.MLP=nn.MLP(7, 2, 32, 1, key=jr.key(1))
		def __call__(self, obs, state, key):
			return self.mlp(obs), None
		def initialize(self, key):
			return None

	mdl = Policy()
	prms, sttcs = eqx.partition(mdl, eqx.is_array)
	fctry = lambda p: eqx.combine(p, sttcs)
	n_sensors = 5
	laser_angles = list(np.linspace(-np.pi/4, np.pi/4, n_sensors))
	tsk = KheperaxTask("standard", False, max_steps=50, model_factory=fctry, robot_kwargs={"laser_angles":laser_angles})
	f, data = tsk(prms, jr.key(3))
	print(f, data["bd"])






