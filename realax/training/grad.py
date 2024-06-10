"""Summary

Attributes:
    Data (TYPE): Description
    Params (TYPE): Description
"""
from .base import BaseTrainer
from ..logging import Logger

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional, Union, Tuple, NamedTuple
from jaxtyping import Float, PyTree

Params = PyTree[...]
Data = PyTree[...]

class TrainState(NamedTuple):
	"""Summary
	"""
	params: Params
	opt_state: optax.OptState
	epoch: int
	best_loss: Float
	best_params: Float

class OptaxTrainer(BaseTrainer):
	
	"""
	Attributes:
	    initializer (TYPE): Description
	    loss_fn (TYPE): Description
	    optimizer (TYPE): Description
	"""
	#-------------------------------------------------------------------
	optimizer: optax.GradientTransformation
	loss_fn: Callable[[Params, jax.Array], Tuple[Float, Data]]
	initializer: Callable[[jax.Array], Params]
	#-------------------------------------------------------------------

	def __init__(
		self, 
		epochs: int,
		optimizer: Union[optax.GradientTransformation, str],
		initializer: Callable[[jax.Array], Params],
		loss_fn: Callable[[Params, jax.Array], Float], 
		learning_rate: Optional[float]=0.01,
		opt_kws: Optional[dict]={},
		logger: Optional[Logger]=None,
	    progress_bar: Optional[bool]=True):
		"""Summary
		
		Args:
		    epochs (int): Description
		    optimizer (Union[optax.GradientTransformation, str]): Description
		    initializer (Callable[[jax.Array], Params]): Description
		    loss_fn (Callable[[Params, jax.Array], Float]): Description
		    learning_rate (Optional[float], optional): Description
		    opt_kws (Optional[dict], optional): Description
		    logger (Optional[Logger], optional): Description
		    progress_bar (Optional[bool], optional): Description
		"""
		super().__init__(epochs, logger=logger, progress_bar=progress_bar)
		
		if isinstance(optimizer, str):
			OPT = getattr(optax, optimizer)
			self.optimizer = OPT(learning_rate=learning_rate, **opt_kws)
		else:
			self.optimizer = optimizer

		self.loss_fn = loss_fn
		self.initializer = initializer

	#-------------------------------------------------------------------

	def train_params(self, params: Params, key: jax.Array)->Tuple[TrainState, Data]:
		"""Summary
		
		Args:
		    params (Params): Description
		    key (jax.Array): Description
		
		Returns:
		    Tuple[TrainState, Data]: Description
		"""
		state = TrainState(params=params, opt_state=self.optimizer.init(params), epoch=0,
						   best_params=params, best_loss=jnp.inf)
		return self.train(state, key) # type: ignore

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jax.Array, data: Optional[PyTree]=None) -> Tuple[TrainState, Data]:
		"""Summary
		
		Args:
		    state (TrainState): Description
		    key (jax.Array): Description
		    data (Optional[PyTree], optional): Description
		
		Returns:
		    Tuple[TrainState, Data]: Description
		"""
		[loss, eval_data], grads = jax.value_and_grad(self.loss_fn, has_aux=True)(state.params, key)
		updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
		params = optax.apply_updates(state.params, updates)
		is_best = loss<state.best_loss
		bl = jnp.where(is_best, loss, state.best_loss)
		bp = jax.tree_map(lambda p, bp: jnp.where(is_best, p, bp), state.params, state.best_params)
		return (TrainState(params=params, opt_state=opt_state, epoch=state.epoch+1, best_params=bp, best_loss=bl), 
				{"loss": loss, "eval_data": eval_data})

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array) -> TrainState:
		"""Summary
		
		Args:
		    key (jax.Array): Description
		
		Returns:
		    TrainState: Description
		"""
		init_params = self.initializer(key)
		opt_state = self.optimizer.init(init_params)
		return TrainState(params=init_params, opt_state=opt_state, epoch=0, 
						  best_params=init_params, best_loss=jnp.inf)


def optimize(
	prms: Params, 
	loss_fn: Callable, 
	key: jax.Array, 
	optimizer: str="adamw", 
	steps: int=1024,
	learning_rate: float=1e-3,
	use_scan=True,
	**kwargs):

	opt = OptaxTrainer(steps, optimizer, initializer=lambda _:prms, loss_fn=loss_fn, 
		learning_rate=learning_rate, **kwargs)

	if use_scan:
		state, data = opt.init_and_train(key)
		return state.params, state, data
	else:
		state = opt.init_and_train_(key)
		return state.params, state
 