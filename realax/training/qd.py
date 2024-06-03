from ctypes import Union
from typing import Callable, TypeAlias, Tuple, Optional, NamedTuple

import jax
import jax.random as jr
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map as shmap
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jaxtyping import Float, PyTree
import evosax as ex
from .base import BaseTrainer
from ..training.log import Logger

from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_euclidean_centroids

Params: TypeAlias = PyTree
Data: TypeAlias = PyTree
RandomKey: TypeAlias = jax.Array
QDTask: TypeAlias = Callable[[Params, RandomKey, Optional[Data]], Tuple[Float, Float, Data]]

class QDState(NamedTuple):
	repertoire: MapElitesRepertoire
	emitter_state: EmitterState


class QDTrainer(BaseTrainer):
	#-------------------------------------------------------------------
	emitter: Emitter
	task : QDTask
	centroids: jax.Array
	params_shaper: ex.ParameterReshaper
	n_devices: int
	fitness_shaper: ex.FitnessShaper
	#-------------------------------------------------------------------

	def __init__(
		self, 
		emitter: Emitter,
		task: QDTask,
		train_steps: int, 
		params_like: Optional[Params]=None,
		param_shaper: Optional[ex.ParameterReshaper]=None,
		grid_shape: Optional[tuple[int]]=None,
		bd_minval: Float=-1.,
		bd_maxval: Float=1.,
		centroids: Optional[jax.Array]=None,
		fitness_shaper: ex.FitnessShaper=ex.FitnessShaper(),
		logger: Optional[Logger] = None, 
		progress_bar: Optional[bool] = False,
		n_devices: int=1):

		super().__init__(train_steps, logger, progress_bar)

		if params_like is None:
			assert param_shaper is not None
			self.params_shaper = param_shaper
		else:
			self.params_shaper = ex.ParameterReshaper(params_like)

		if grid_shape is None:
			assert centroids is not None
			self.centroids = centroids
		else: 
			self.centroids = compute_euclidean_centroids(grid_shape, bd_minval, bd_maxval)


		self.emitter = emitter
		self.task = task
		self.n_devices = n_devices
		self.fitness_shaper = fitness_shaper

	#-------------------------------------------------------------------

	def train_step(self, state: QDState, key: RandomKey, data: Optional[Data] = None) -> Tuple[QDState, Data]:
		
		kemit, keval = jr.split(key, 2)
		x, _ = self.emitter.emit(state.repertoire, state.emitter_state, kemit)
		fitness, bd, eval_data = self.eval(x, keval, data) 
		f = self.fitness_shaper.apply(x, fitness)
		repertoire = state.repertoire.add(
			x, bd, f, eval_data
		)
		emitter_state = self.emitter.state_update(
			emitter_state=state.emitter_state,
			repertoire=state.repertoire,
			genotypes=x,
			fitnesses=f,
			descriptors=bd,
			extra_scores=eval_data,
		)
		state = state._replace(repertoire=repertoire, emitter_state=emitter_state)
		return state, dict(fitness=fitness, bd=bd, eval_data=eval_data)

	#-------------------------------------------------------------------

	def initialize(self, key: jax.Array) -> QDState:
		return super().initialize(key)

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

	def _eval(self, x: jax.Array, key: RandomKey, data: Data):
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

	def _eval_shmap(self, x: jax.Array, key: jax.Array, data: Data):
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

		_eval = lambda x, k: self.task(self.params_shaper.reshape_single(x), k, None)
		batch_eval = jax.vmap(_eval, in_axes=(0,None,None))
		sheval = shmap(batch_eval, 
					   mesh=device_mesh, 
					   in_specs=(P("p",), P(), P()),
					   out_specs=(P("p"), P("p"), P("p")),
					   check_rep=False)

		return sheval(x, key, data)