"""Summary

Deleted Attributes:
    Data (TYPE): Description
    Params (TYPE): Description
    Task (TYPE): Description
    TaskParams (TYPE): Description
    TrainState (TYPE): Description
"""
from ...logging import Logger
from ..base import BaseTrainer

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental.shard_map import shard_map as shmap
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import evosax as ex
from typing import Any, Callable, Dict, Optional, TypeAlias, Union, Tuple
from jaxtyping import Float, PyTree

TrainState: TypeAlias  = ex.EvoState
Params: TypeAlias  = PyTree
Data: TypeAlias  = PyTree
TaskParams: TypeAlias  = PyTree
Task: TypeAlias  = Callable
RandomKey: TypeAlias = jax.Array

class EvosaxTrainer(BaseTrainer):
	
	"""
	Attributes:
	    es_params (TYPE): Description
	    fitness_shaper (TYPE): Description
	    n_devices (TYPE): Description
	    params_shaper (TYPE): Description
	    strategy (TYPE): Description
	    task (TYPE): Description
	
	Deleted Attributes:
	    multi_device_mode (TYPE): Description
	"""
	#-------------------------------------------------------------------
	strategy: ex.Strategy
	es_params: ex.EvoParams
	params_shaper: ex.ParameterReshaper
	task: Task
	fitness_shaper: ex.FitnessShaper
	n_devices: int
	#-------------------------------------------------------------------

	def __init__(
		self, 
		train_steps: int,
		strategy: Union[ex.Strategy, str],
		task: Callable[[Params, RandomKey, Data], Tuple[Float,Data]],
		params_like: Optional[Params]=None,
		params_shaper: Optional[ex.ParameterReshaper]=None,
		popsize: Optional[int]=None,
		fitness_shaper: Optional[ex.FitnessShaper]=None,
		es_kws: Optional[Dict[str, Any]]={},
		es_params: Optional[ex.EvoParams]=None,
		eval_reps: int=1,
		logger: Optional[Logger]=None,
	    progress_bar: Optional[bool]=False,
	    n_devices: int=1):
		"""Summary
		
		Args:
		    train_steps (int): number of training steps (generations)
		    strategy (Union[ex.Strategy, str]): string specifying one of evosax implemented strategies or directly a 
		    	strategy following evosax API
		    task (Callable[[Params, RandomKey, Data], Tuple[Float, Data]]): task (i.e fitness function)
		    params_like (Optional[Params], optional): Tree definition of parameters to be evolved 
		    	(if not specified, a ParameterReshaper must be given as 'params_shaper')
		    params_shaper (Optional[ex.ParameterReshaper], optional): Description
		    popsize (Optional[int], optional): population size
		    fitness_shaper (Optional[ex.FitnessShaper], optional): fitness modificator, by default fitness is minimized
		    es_kws (Optional[Dict[str, Any]], optional): keyword arguments to be passer to the es if specified as a string in 'strategy'
		    es_params (Optional[ex.EvoParams], optional): optional parameters for the strategy
		    eval_reps (int, optional): number of evaluations per individual (Monte-carlo trials, useful only if task is stochastic)
		    logger (Optional[Logger], optional): optional logger
		    progress_bar (Optional[bool], optional): specifies if a progress bar is displayed during training
		    n_devices (int, optional): number of devices to parallelize training over
		"""
		super().__init__(train_steps=train_steps, 
						 logger=logger, 
						 progress_bar=progress_bar)

		if params_like is None:
			assert params_shaper is not None, "one of params_like or params_shaper must be given"
			self.params_shaper = params_shaper
		else:
			self.params_shaper = ex.ParameterReshaper(params_like)
		
		if isinstance(strategy, str):
			assert popsize is not None
			self.strategy = self.create_strategy(strategy, popsize, self.params_shaper.total_params, **es_kws) # type: ignore
		else:
			self.strategy = strategy

		if es_params is None:
			self.es_params = self.strategy.default_params
		else:
			self.es_params = es_params

		if eval_reps > 1:
			def _eval_fn(p: Params, k: jax.Array, tp: Optional[PyTree]=None):
				fit, info = jax.vmap(task, in_axes=(None,0,None))(p, jr.split(k,eval_reps), tp)
				return jnp.mean(fit), info
			self.task = _eval_fn
		else :
			self.task = task

		if fitness_shaper is None:
			self.fitness_shaper = ex.FitnessShaper()
		else:
			self.fitness_shaper = fitness_shaper

		self.n_devices = n_devices

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jax.Array, data: Optional[TaskParams]=None) -> Tuple[TrainState, Data]:
		"""Summary
		
		Args:
		    state (TrainState): Description
		    key (jax.Array): Description
		    data (Optional[TaskParams], optional): Description
		
		Returns:
		    Tuple[TrainState, Data]: Description
		"""
		ask_key, eval_key = jr.split(key, 2)
		x, state = self.strategy.ask(ask_key, state, self.es_params)
		fitness, eval_data = self.eval(x, eval_key, data)
		f = self.fitness_shaper.apply(x, fitness)
		state = self.strategy.tell(x, f, state, self.es_params)
		return state, {"fitness": fitness, "eval_data": eval_data} #TODO def best as >=

	#-------------------------------------------------------------------

	def eval(self, *args, **kwargs):
		"""Summary
		
		Args:
		    *args: Description
		    **kwargs: Description
		
		Returns:
		    TYPE: Description
		
		No Longer Raises:
		    ValueError: Description
		"""
		if self.n_devices == 1:
			return self._eval(*args, **kwargs)
		else:
			return self._eval_shmap(*args, **kwargs)

	#-------------------------------------------------------------------

	def _eval(self, x: jax.Array, key: RandomKey, data: Data)->Tuple[jax.Array, PyTree]:
		"""Summary
		
		Args:
		    x (jax.Array): Description
		    key (RandomKey): Description
		    data (Data): Description
		
		Returns:
		    Tuple[jax.Array, PyTree]: Description
		
		Deleted Parameters:
		    task_params (PyTree): Description
		"""
		params = self.params_shaper.reshape(x)
		_eval = jax.vmap(self.task, in_axes=(0, 0, None))
		return _eval(params, jr.split(key, x.shape[0]), data)

	#-------------------------------------------------------------------

	def _eval_shmap(self, x: jax.Array, key: jax.Array, data: Data)->Tuple[jax.Array, PyTree]:
		"""Summary
		
		Args:
		    x (jax.Array): Description
		    key (jax.Array): Description
		    task_params (PyTree): Description
		
		Returns:
		    Tuple[jax.Array, PyTree]: Description
		"""
		devices = mesh_utils.create_device_mesh((self.n_devices,))
		device_mesh = Mesh(devices, axis_names=("p"))

		_eval = lambda x, k, d: self.task(self.params_shaper.reshape_single(x), k, d)
		batch_eval = jax.vmap(_eval, in_axes=(0,None,None))
		sheval = shmap(batch_eval, 
					   mesh=device_mesh, 
					   in_specs=(P("p",), P(), P()),
					   out_specs=(P("p"), P("p")),
					   check_rep=False)

		return sheval(x, key, data)

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array, **kwargs) -> TrainState:
		"""Summary
		
		Args:
		    key (jax.Array): Description
		    **kwargs: Description
		
		Returns:
		    TrainState: Description
		"""
		state = self.strategy.initialize(key, self.es_params)
		state = state.replace(**kwargs)
		return state

	#-------------------------------------------------------------------

	def create_strategy(self, name: str, popsize: int, num_dims: int, **kwargs)->ex.Strategy:
		"""Summary
		
		Args:
		    name (str): Description
		    popsize (int): Description
		    num_dims (int): Description
		    **kwargs: Description
		
		Returns:
		    ex.Strategy: Description
		"""
		ES = getattr(ex, name)
		es = ES(popsize=popsize, num_dims=num_dims, **kwargs)
		return es

	#-------------------------------------------------------------------



def evolve(
	prms: Params, 
	fitness_fn: Callable, 
	key: jax.Array, 
	popsize: int=64, 
	strategy:Union[ex.Strategy, str] ="OpenES", 
	steps: int=1024, 
	use_scan: bool=True, 
	**kwargs)->Tuple[Params,ex.EvoState,Data]|Tuple[Params,ex.EvoState]:
	"""handy function to quickly evolve some parameters wrt a fitness function
	
	Args:
	    prms (Params): tree definition of parameters to evolve
	    fitness_fn (Callable): fitness function 
	    key (jax.Array): rng key
	    popsize (int, optional): population size
	    strategy (Union[ex.Strategy, str], optional): es strategy to use
	    steps (int, optional): number of training steps
	    use_scan (bool, optional): wether training loop is done with as scan or fori_loop
	    **kwargs: optional keyword args to be passed to EvosaxTrainer (e.g logger, fitness_shaper)
	
	Returns:
	    Tuple[Params, ex.EvoState, Data] | Tuple[Params, ex.EvoState]
	"""
	ps = ex.ParameterReshaper(prms, verbose=False)
	es = EvosaxTrainer(steps, strategy, fitness_fn, params_like=prms, popsize=popsize, **kwargs)
	if use_scan:
		state, data = es.init_and_train(key)
		evolved_prms = ps.reshape_single(state.mean)
		return evolved_prms, state, data
	else:
		state = es.init_and_train_(key)
		evolved_prms = ps.reshape_single(state.mean)
		return evolved_prms, state










