"""Summary
"""
#TODO: add option to add data generator (default return None)
# providing data ate each trainign step

from .utils import progress_bar_scan, progress_bar_fori
from ..logging import Logger

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Optional, Tuple, Any, TypeAlias
import jax.experimental.host_callback as hcb
import wandb
from jaxtyping import PyTree

Data: TypeAlias = PyTree[...]
TrainState: TypeAlias = PyTree[...]

class BaseTrainer:

	def __init__(self, 
				 train_steps: int, 
				 logger: Optional[Logger]=None,
				 progress_bar: Optional[bool]=False):
		"""
		Base trainer class: implements trianing loops and data management
		
		Args:
		    train_steps (int): number of training steps
		    logger (Optional[Logger], optional): optional logger
		    progress_bar (Optional[bool], optional): if progress bar is dispalyed during training
		"""
		
		self.train_steps = train_steps
		self.progress_bar = progress_bar
		self.logger = logger

	#-------------------------------------------------------------------

	def __call__(self, key: jax.Array):
		"""Callback for init_and_train
		
		Args:
		    key (jax.Array)
		
		Returns:
		    TYPE: Description
		"""
		return self.init_and_train(key)

	#-------------------------------------------------------------------

	def train(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->Tuple[TrainState, Data]:
		"""
		Executes training from initial training state using scan
		Since scan stacks all the outputs this method can require a lot of memory 
		(see train_ for a lighter method)
		
		Args:
		    state (TrainState): Description
		    key (jax.Array): Description
		    data (Optional[Data], optional): Description
		
		Returns:
		    Tuple[TrainState, Data]: Returns final train state as well as stacked data
		"""

		def _step(c, x):
			"""Summary
			
			Args:
			    c (TYPE): Description
			    x (TYPE): Description
			
			Returns:
			    TYPE: Description
			"""
			s, k = c
			k, k_ = jr.split(k)
			s, step_data = self.train_step(s, k_, data)
			
			if self.logger is not None:
				self.logger.log(s, step_data)

			return [s, k], {"states": s, "metrics": step_data}

		if self.progress_bar:
			_step = progress_bar_scan(self.train_steps)(_step) #type: ignore

		[state, key], train_data = jax.lax.scan(_step, [state, key], jnp.arange(self.train_steps))

		return state, train_data

	#-------------------------------------------------------------------

	def train_(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->TrainState:
		"""
		Executes training from initial training state using fori_loop
		
		Args:
		    state (TrainState): Description
		    key (jax.Array): Description
		    data (Optional[Data], optional): Description
		
		Returns:
		    TrainState: Description
		"""

		def _step(i, c):
			"""Summary
			
			Args:
			    i (TYPE): Description
			    c (TYPE): Description
			
			Returns:
			    TYPE: Description
			"""
			s, k = c
			k, k_ = jr.split(k)
			s, step_data = self.train_step(s, k_, data)
			if self.logger is not None:
				self.logger.log(s, step_data)
			return [s, k]

		if self.progress_bar:
			_step = progress_bar_fori(self.train_steps)(_step) #type: ignore

		[state, key] = jax.lax.fori_loop(0, self.train_steps, _step, [state, key])
		return state

	#-------------------------------------------------------------------

	def init_and_train(self, key: jax.Array, data: Optional[Data]=None)->Tuple[TrainState, Data]:
		"""Summary
		
		Args:
		    key (jax.Array): Description
		    data (Optional[Data], optional): Description
		
		Returns:
		    Tuple[TrainState, Data]: Description
		"""
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train(state, train_key, data)

	#-------------------------------------------------------------------

	def init_and_train_(self, key: jax.Array, data: Optional[Data]=None)->TrainState:
		"""Summary
		
		Args:
		    key (jax.Array): Description
		    data (Optional[Data], optional): Description
		
		Returns:
		    TrainState: Description
		"""
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train_(state, train_key, data)

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->Tuple[TrainState, Any]:
		"""Summary
		
		Args:
		    state (TrainState): Description
		    key (jax.Array): Description
		    data (Optional[Data], optional): Description
		
		Raises:
		    NotImplementedError: Description
		"""
		raise NotImplementedError

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array)->TrainState:
		"""Summary
		
		Args:
		    key (jax.Array): Description
		
		Raises:
		    NotImplementedError: Description
		"""
		raise NotImplementedError
