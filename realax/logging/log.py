"""Summary
"""
import wandb
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from typing import Tuple, TypeAlias, Callable, Optional
import equinox as eqx
import os
from jax.experimental import io_callback

TrainState: TypeAlias = PyTree[...]
Data: TypeAlias = PyTree[...]

def default_es_metrics(state, data):
	"""Summary
	
	Args:
	    state (TYPE): Description
	    data (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	gen = state.gen_counter
	f = data["fitness"]
	log_data = dict(
		avg_fitness=f.mean(),
		min_fitness=f.min(),
		max_fitness=f.max()
	)
	ckpt_data = state.mean
	return log_data, ckpt_data, gen

def default_grad_metrics(state, data):
	"""Summary
	
	Args:
	    state (TYPE): Description
	    data (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	ckpt_data = state.params
	epoch = state.epoch
	log_data = data["loss"]
	return log_data, ckpt_data, epoch



class Logger:

	"""Summary
	
	Attributes:
	    ckpt_file (TYPE): Description
	    ckpt_freq (TYPE): Description
	    host_log_transform (TYPE): Description
	    metrics_fn (TYPE): Description
	    verbose (TYPE): Description
	    wandb_log (TYPE): Description
	"""
	
	#-------------------------------------------------------------------

	def __init__(
		self, 
		wandb_log: bool,
		metrics_fn: Callable[[TrainState, Data], Tuple[Data, Data, int]],
		host_log_transform: Callable[[Data], Data]=lambda x:x,
		ckpt_file: Optional[str]=None, 
		ckpt_freq: int=100,
		verbose: bool=False):
		"""Logging class
		
		Args:
		    wandb_log (bool): if data is to be logged on wandb
		    metrics_fn (Callable[[TrainState, Data], Tuple[Data, Data, int]]): function taking 
		    	as input the current train state and data from evaluation and returning a three tuple
		    	being : (dict to log on wandb, data to save as ckpt, current trainint iteration)
		    host_log_transform (Callable[[Data], Data], optional): function run on host side optionally t
		    	ransforming the data passed to log on wandb 
		    ckpt_file (Optional[str], optional): file prefix where to save ckpts
		    ckpt_freq (int, optional): frequency at which to save ckpts
		    verbose (bool, optional): Description
		"""
		if ckpt_file is not None and "/" in ckpt_file:
			if not os.path.isdir(ckpt_file[:ckpt_file.rindex("/")]):
				os.makedirs(ckpt_file[:ckpt_file.rindex("/")])
		self.wandb_log = wandb_log
		self.metrics_fn = metrics_fn
		self.ckpt_file = ckpt_file
		self.ckpt_freq = ckpt_freq
		self.verbose = verbose
		self.host_log_transform = host_log_transform

	#-------------------------------------------------------------------

	def log(self, state: TrainState, data: Data):
		"""Summary
		
		Args:
		    state (TrainState): Description
		    data (Data): Description
		
		Returns:
		    TYPE: Description
		"""
		log_data, ckpt_data, epoch = self.metrics_fn(state, data)
		if self.wandb_log:
			self._log(log_data)
		if self.ckpt_file is not None:
			self.save_ckpt(ckpt_data, epoch)
		return log_data

	#-------------------------------------------------------------------

	def _log(self, data: dict):
		"""Summary
		
		Args:
		    data (dict): Description
		"""
		def clbck(data, *_):
			wandb.log(self.host_log_transform(data))
			return jnp.zeros((),dtype=bool)

		_ = io_callback(clbck, jax.ShapeDtypeStruct((),bool), data)

	#-------------------------------------------------------------------

	def save_ckpt(self, data: dict, epoch: int):
		"""Summary
		
		Args:
		    data (dict): Description
		    epoch (int): Description
		"""
		def save(data, epoch):
			"""Summary
			
			Args:
			    data (TYPE): Description
			    epoch (TYPE): Description
			"""
			assert self.ckpt_file is not None
			file = f"{self.ckpt_file}_{int(epoch)}.eqx"
			if self.verbose:
				print("saving data at: ", file)
			eqx.tree_serialise_leaves(file, data)

		def tap_save(data, epoch):
			"""Summary
			
			Args:
			    data (TYPE): Description
			    epoch (TYPE): Description
			
			Returns:
			    TYPE: Description
			"""
			hcb.id_tap(lambda de, _: save(de[0], de[1]), [data,epoch])
			return None

		if self.ckpt_file is not None:
			jax.lax.cond(
				(jnp.mod(epoch, self.ckpt_freq))==0,
				lambda data, epoch : tap_save(data, epoch),
				lambda data, epoch : None,
				data, epoch
			)

	#-------------------------------------------------------------------

	def init(self, project: str, config: dict, **kwargs):
		"""Summary
		
		Args:
		    project (str): Description
		    config (dict): Description
		    **kwargs: Description
		"""
		if self.wandb_log:
			wandb.init(project=project, config=config, **kwargs)

	#-------------------------------------------------------------------

	def finish(self, *args, **kwargs):
		"""Summary
		
		Args:
		    *args: Description
		    **kwargs: Description
		"""
		if self.wandb_log:
			wandb.finish(*args, **kwargs)
